# SPDX-License-Identifier: Apache-2.0
"""Per-worker steering state and forward-pre-hook installation.

Design:
    * One ``SteerState`` singleton per worker process.
    * ``install_steer_hooks(model)`` walks the model, finds decoder-layer
      modules, tags each with its layer index, and registers a single
      forward-pre-hook on all of them.
    * ``SteerState.set_active(request)`` materialises scaled vectors on the
      GPU and populates a {layer_idx: tensor} map that the hook consults.
    * ``SteerState.clear()`` empties the map → hook becomes a no-op that
      returns ``None`` (PyTorch runs original forward with untouched args).

This gives byte-identical nosteer output because the hook's fast path is a
single dict lookup followed by an early return — no tensor allocation, no
recombination of (hidden_states, residual).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch import nn

from vllm.steer.request import SteerVectorRequest

logger = logging.getLogger(__name__)

_STATE: "SteerState | None" = None
_LAYER_IDX_ATTR = "_vllm_steer_layer_idx"

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)$"),
    re.compile(r"(?:^|\.)h\.(\d+)$"),
    re.compile(r"(?:^|\.)blocks\.(\d+)$"),
)
_SKIP_SUBSTRINGS = ("vision", "visual", "image", "audio", "speech")


def get_steer_state() -> "SteerState":
    global _STATE
    if _STATE is None:
        _STATE = SteerState()
    return _STATE


class SteerState:
    """Per-worker mutable steering state.

    Attributes
    ----------
    active:
        Mapping ``layer_idx -> scaled_vector`` for the current forward pass.
        Empty when no steering is active (hook fast-path).
    hook_mode:
        ``"pre"`` (add to layer input — HF semantics, ~3× stronger) or
        ``"post"`` (add to layer output — old vllm-steer semantics, used for
        reproducing the reference Gate-2 numbers).
    """

    def __init__(self) -> None:
        self.active: dict[int, torch.Tensor] = {}
        self.hook_mode: str = "pre"
        self._cache: dict[tuple, torch.Tensor] = {}
        self._current_key: tuple | None = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._n_layers: int = 0
        self._layers: list[nn.Module] = []
        # Capture mode: record residual-stream activations instead of steering.
        # Used for direction extraction via vLLM (bypasses HF memory issues).
        self.capture_mode: bool = False
        self.capture_positions: list[int] | None = None  # None = last token only
        self.captured: dict[int, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # Hook installation                                                  #
    # ------------------------------------------------------------------ #

    def install_hooks(self, model: nn.Module) -> int:
        """Find decoder-layer modules and register the steering pre-hook on each.

        Returns the number of layers instrumented.
        """
        if self._handles:
            return self._n_layers  # idempotent

        layers: list[tuple[int, str, nn.Module]] = []
        for name, module in model.named_modules():
            if any(s in name for s in _SKIP_SUBSTRINGS):
                continue
            for pat in _LAYER_PATTERNS:
                m = pat.search(name)
                if m:
                    layers.append((int(m.group(1)), name, module))
                    break

        if not layers:
            raise RuntimeError(
                "install_steer_hooks: no decoder layers found. "
                "Searched for modules matching 'layers.N', 'h.N', 'blocks.N'."
            )

        layers.sort(key=lambda t: t[0])
        example_param = next(model.parameters())
        self._device = example_param.device
        self._dtype = example_param.dtype
        self._layers = []
        for idx, name, module in layers:
            setattr(module, _LAYER_IDX_ATTR, idx)
            self._handles.append(
                module.register_forward_pre_hook(
                    _steer_pre_hook, with_kwargs=True
                )
            )
            self._handles.append(
                module.register_forward_hook(
                    _steer_post_hook, with_kwargs=True
                )
            )
            self._layers.append(module)
        self._n_layers = len(layers)
        logger.info(
            "steer: installed hooks on %d decoder layers (device=%s dtype=%s) "
            "— example: %s",
            self._n_layers,
            self._device,
            self._dtype,
            layers[0][1],
        )
        return self._n_layers

    # ------------------------------------------------------------------ #
    # Activation                                                         #
    # ------------------------------------------------------------------ #

    def set_active(self, req: SteerVectorRequest | None) -> None:
        """Set (or clear) the active steering configuration for the next
        model-forward call(s). Safe to call redundantly — it no-ops if the
        config hasn't changed."""
        if req is None:
            if self._current_key is not None:
                self.active = {}
                self._current_key = None
            return

        key = req.config_key
        if key == self._current_key:
            return  # already active, nothing to do

        direction = self._load_direction(req)
        targets = req.target_layers
        if not targets:
            targets = list(range(min(self._n_layers, direction.shape[0])))

        active: dict[int, torch.Tensor] = {}
        for layer in targets:
            if layer < 0 or layer >= direction.shape[0]:
                raise ValueError(
                    f"target_layer {layer} out of range for direction tensor "
                    f"with {direction.shape[0]} layers"
                )
            vec = direction[layer] * req.scale
            if req.normalize:
                vec = vec / (vec.norm() + 1e-8)
            active[layer] = vec.to(device=self._device, dtype=self._dtype)
        self.active = active
        self.hook_mode = req.hook
        self._current_key = key
        logger.info(
            "steer: activated %r scale=%.3f hook=%s layers=%s",
            req.steer_vector_name,
            req.scale,
            req.hook,
            sorted(active.keys()),
        )

    def clear(self) -> None:
        self.set_active(None)

    # ------------------------------------------------------------------ #
    # Activation capture (for direction extraction via vLLM)             #
    # ------------------------------------------------------------------ #

    def start_capture(self, positions: list[int] | None = None) -> None:
        """Enable capture mode. The pre-hook will record residual-stream
        activations at ``positions`` (negative indexing from sequence end;
        None = last token only) for every layer on every forward pass."""
        self.capture_mode = True
        self.capture_positions = positions or [-1]
        self.captured = {i: [] for i in range(self._n_layers)}
        logger.info(
            "steer: capture mode ON (positions=%s, %d layers)",
            self.capture_positions, self._n_layers,
        )

    def stop_capture(self) -> dict[int, torch.Tensor]:
        """Disable capture and return the recorded activations as a dict of
        {layer_idx: Tensor[n_forwards, n_positions, d_model]}."""
        self.capture_mode = False
        out = {}
        for layer_idx, tensors in self.captured.items():
            if tensors:
                out[layer_idx] = torch.stack(tensors, dim=0).cpu()
        self.captured = {}
        logger.info("steer: capture mode OFF (%d layers recorded)", len(out))
        return out

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _load_direction(self, req: SteerVectorRequest) -> torch.Tensor:
        ck = req.cache_key
        if ck in self._cache:
            return self._cache[ck]
        t = torch.load(
            req.steer_vector_local_path, map_location="cpu", weights_only=True
        )
        if not isinstance(t, torch.Tensor):
            raise ValueError(
                f"steer: {req.steer_vector_local_path} does not contain a "
                f"tensor (got {type(t).__name__})"
            )
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2:
            raise ValueError(
                f"steer: direction tensor must be 1-D [d_model] or 2-D "
                f"[n_layers, d_model], got shape {tuple(t.shape)}"
            )
        self._cache[ck] = t
        logger.info(
            "steer: loaded direction %r from %s shape=%s dtype=%s",
            req.steer_vector_name,
            req.steer_vector_local_path,
            tuple(t.shape),
            t.dtype,
        )
        return t


# ---------------------------------------------------------------------- #
# Hook functions                                                         #
# ---------------------------------------------------------------------- #


def _find_residual(
    args: tuple, kwargs: dict
) -> tuple[Any, bool, int | str | None]:
    """Locate ``hidden_states`` / ``residual`` among positional+keyword args.

    vLLM decoder-layer forwards take (positions, hidden_states, residual, …)
    positionally from the model loop. We add the steering vector to ``residual``
    (or to ``hidden_states`` if residual is None — the first layer).

    Returns (tensor_to_modify, is_kwarg, index_or_key). For a non-None residual
    we return the residual; otherwise we return hidden_states.
    """
    if "residual" in kwargs:
        r = kwargs["residual"]
        if r is not None:
            return r, True, "residual"
        # fall through → find hidden_states
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"], True, "hidden_states"
        # hidden_states positional? (unlikely)
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor) and a.dim() >= 2:
                return a, False, i
        return None, False, None

    # Typical case: all positional. Standard layout is
    #   args = (positions, hidden_states, residual, *rest)
    if len(args) >= 3 and (args[2] is None or isinstance(args[2], torch.Tensor)):
        if args[2] is not None:
            return args[2], False, 2
        return args[1], False, 1

    # Fallback: last 2-D tensor that isn't positions (positions is 1-D).
    for i in range(len(args) - 1, -1, -1):
        a = args[i]
        if isinstance(a, torch.Tensor) and a.dim() >= 2:
            return a, False, i
    return None, False, None


def _steer_pre_hook(module: nn.Module, args: tuple, kwargs: dict):
    """Forward-pre-hook: add steering vector to the residual-stream INPUT
    of this decoder layer (HF ``register_forward_pre_hook`` semantics).
    Also supports capture mode for vLLM-based direction extraction."""
    state = _STATE
    if state is None:
        return None
    idx = getattr(module, _LAYER_IDX_ATTR, None)
    if idx is None:
        return None

    # Capture mode: record activations, don't steer.
    if state.capture_mode:
        tgt, _, _ = _find_residual(args, kwargs)
        if tgt is not None and tgt.dim() >= 2:
            # tgt shape: [seq_len, d_model] or [batch*seq, d_model]
            # Record the specified positions (last-N tokens in the flattened seq).
            positions = state.capture_positions or [-1]
            try:
                captured = tgt[positions].detach().clone()  # [n_pos, d_model]
                state.captured.setdefault(idx, []).append(captured)
            except (IndexError, RuntimeError):
                pass  # sequence too short for requested positions
        return None

    if not state.active or state.hook_mode != "pre":
        return None
    vec = state.active.get(idx)
    if vec is None:
        return None

    tgt, is_kwarg, loc = _find_residual(args, kwargs)
    if tgt is None:
        return None

    # Out-of-place add so the caller's reference isn't mutated — avoids
    # surprises under pipeline parallelism / aux-hidden-state capture.
    new = tgt + vec.to(device=tgt.device, dtype=tgt.dtype)
    if is_kwarg:
        new_kwargs = dict(kwargs)
        new_kwargs[loc] = new
        return args, new_kwargs
    new_args = list(args)
    new_args[loc] = new
    return tuple(new_args), kwargs


def _steer_post_hook(module: nn.Module, args: tuple, kwargs: dict, output):
    """Forward-post-hook: add steering vector to the layer OUTPUT (old
    vllm-steer semantics — post-hook at layer L ≈ pre-hook at L+1).

    Unlike the old fork this does NOT collapse (h, r) → (h+r, 0). We add to
    ``residual`` in the output tuple, leaving ``hidden_states`` untouched,
    which is numerically indistinguishable from adding to the combined stream
    (the next layer's fused RMSNorm sums h+r anyway).
    """
    state = _STATE
    if state is None or not state.active or state.hook_mode != "post":
        return None
    idx = getattr(module, _LAYER_IDX_ATTR, None)
    if idx is None:
        return None
    vec = state.active.get(idx)
    if vec is None:
        return None

    if not isinstance(output, tuple) or len(output) < 2:
        return None
    h, r, *rest = output
    if r is None or not isinstance(r, torch.Tensor):
        # No residual split → add to hidden_states directly.
        if isinstance(h, torch.Tensor):
            return (h + vec.to(device=h.device, dtype=h.dtype), r, *rest)
        return None
    return (h, r + vec.to(device=r.device, dtype=r.dtype), *rest)


# Public alias for the module-level install call.
def install_steer_hooks(model: nn.Module) -> int:
    return get_steer_state().install_hooks(model)
