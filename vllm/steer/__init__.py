# SPDX-License-Identifier: Apache-2.0
"""Activation steering for vLLM.

Per-request activation-addition steering via forward pre-hooks on decoder
layers. Hook semantics match the HuggingFace gold-standard
(register_forward_pre_hook on the decoder block): the steering vector is
added to the layer *input* before attention/MLP run, so the perturbation
propagates through the layer's own computation.

Key invariant: when no steer vector is active, the hook returns ``None`` —
PyTorch then runs the original forward with untouched args. No tensor
operations, no (hidden_states, residual) recombination, so the nosteer path
is byte-identical to stock vLLM.
"""

from vllm.steer.request import SteerVectorRequest
from vllm.steer.state import (
    SteerState,
    get_steer_state,
    install_steer_hooks,
)

__all__ = [
    "SteerVectorRequest",
    "SteerState",
    "get_steer_state",
    "install_steer_hooks",
]
