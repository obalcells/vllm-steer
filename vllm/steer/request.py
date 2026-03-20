# SPDX-License-Identifier: Apache-2.0
"""Request dataclass for per-request activation steering."""

from __future__ import annotations

import msgspec


class SteerVectorRequest(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):
    """Steering-vector spec attached to a single generation request.

    The server loads ``steer_vector_local_path`` (a ``.pt`` file on the
    server's local filesystem) and adds ``scale * direction[layer]`` to the
    residual-stream input of each ``target_layers[i]`` decoder block.
    """

    steer_vector_name: str
    steer_vector_int_id: int
    steer_vector_local_path: str = ""
    scale: float = 1.0
    target_layers: list[int] | None = None
    prefill_trigger_tokens: list[int] | None = None
    generate_trigger_tokens: list[int] | None = None
    algorithm: str = "direct"
    normalize: bool = False
    hook: str = "pre"

    def __post_init__(self) -> None:
        if self.steer_vector_int_id < 1:
            raise ValueError(
                f"steer_vector_int_id must be > 0, got {self.steer_vector_int_id}"
            )
        if not self.steer_vector_local_path:
            raise ValueError("steer_vector_local_path is required")
        if self.algorithm not in ("direct",):
            raise ValueError(
                f"algorithm must be 'direct', got {self.algorithm!r}"
            )
        if self.hook not in ("pre", "post"):
            raise ValueError(f"hook must be 'pre' or 'post', got {self.hook!r}")

    @property
    def cache_key(self) -> tuple:
        """Key for caching loaded direction tensors across requests."""
        return (self.steer_vector_int_id, self.steer_vector_local_path)

    @property
    def config_key(self) -> tuple:
        """Key for detecting when the active steer config changes between
        consecutive model-forward calls (if unchanged, skip re-broadcast)."""
        return (
            self.steer_vector_int_id,
            self.steer_vector_local_path,
            self.scale,
            tuple(self.target_layers or ()),
            tuple(self.prefill_trigger_tokens or ()),
            tuple(self.generate_trigger_tokens or ()),
            self.normalize,
            self.hook,
        )

    def applies_to_prefill(self) -> bool:
        toks = self.prefill_trigger_tokens
        return bool(toks) and (toks == [-1] or -1 in toks)

    def applies_to_generate(self) -> bool:
        toks = self.generate_trigger_tokens
        return bool(toks) and (toks == [-1] or -1 in toks)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SteerVectorRequest)
            and self.config_key == other.config_key
        )

    def __hash__(self) -> int:
        return hash(self.config_key)
