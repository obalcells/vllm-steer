# vllm-steer2: Activation Steering for vLLM

Minimal fork of vLLM v0.15.0 that adds per-request activation-addition
steering via the OpenAI-compatible API.

## Design

Steering is implemented as PyTorch `forward_pre_hook`s on decoder-layer
modules. When active, the hook adds `scale * direction[layer]` to the
residual-stream **input** of each target layer (HF gold-standard semantics).

**Key invariant:** when no steer vector is active, the hook returns `None`
and PyTorch runs the original forward with untouched args — nosteer output
is **byte-identical** to stock vLLM. No wrapper class, no `(h,r)→(h+r,0)`
collapse, no timing perturbation under TP.

## Install

```bash
# 1. Have stock vllm==0.15.0 installed (for compiled kernels)
# 2. Clone this fork and link precompiled artifacts
git clone <this-repo> vllm-steer2
cd vllm-steer2
./link_precompiled.sh    # symlinks .so kernels + generated files from wheel

# 3. Shadow-install via .pth (no build needed)
echo 'import sys; sys.path.insert(0, "'$PWD'")' \
    > $(python -c 'import site; print(site.getsitepackages()[0])')/\_vllm_steer.pth
```

The fork shadows the installed wheel's Python files while reusing its
compiled kernels. `link_precompiled.sh` is **required** — the git tree
lacks `third_party/triton_kernels/` etc and vLLM silently falls back to
numerically-different kernel implementations without them.

## Usage

### Start server

```bash
vllm serve <model-path> \
    --served-model-name my-model \
    --enforce-eager \            # REQUIRED: forward hooks need eager mode
    --no-enable-prefix-caching \ # recommended: prefix KVs may be stale after steering
    --tensor-parallel-size N \
    --port 8000 --api-key inspectai
```

### Set steering (HTTP endpoint)

```bash
curl -X POST http://localhost:8000/v1/steer_vector \
    -H "Authorization: Bearer inspectai" \
    -H "Content-Type: application/json" \
    -d '{
        "steer_vector_name": "eval_awareness",
        "steer_vector_int_id": 1,
        "steer_vector_local_path": "/path/to/direction.pt",
        "scale": -0.55,
        "target_layers": [16, 19, 22, 25, 28],
        "hook": "pre"
    }'

# clear
curl -X DELETE http://localhost:8000/v1/steer_vector \
    -H "Authorization: Bearer inspectai"
```

### Per-request steering (OpenAI API)

```python
from openai import OpenAI
cli = OpenAI(base_url="http://localhost:8000/v1", api_key="inspectai")

r = cli.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"steer_vector": {
        "steer_vector_name": "ea",
        "steer_vector_int_id": 1,
        "steer_vector_local_path": "/path/to/direction.pt",
        "scale": -0.55,
        "target_layers": [16, 19, 22, 25, 28],
        "hook": "pre",
    }}
)
```

Steering is **global across concurrent requests** — the first request with
a `steer_vector` broadcasts it to all workers via `collective_rpc`.
Subsequent requests with the same config are no-ops. Send `steer_vector=None`
(or omit it) and DELETE `/v1/steer_vector` to clear.

## Direction tensor format

`.pt` file containing a 2-D tensor `[n_layers, d_model]` (or 1-D `[d_model]`
applied to all target layers). The server loads it on first use and indexes
`direction[layer]` for each `target_layers[i]`.

## Hook modes

- `hook: "pre"` (default) — add to layer **input** before attn/MLP. Matches
  HF `register_forward_pre_hook`. Perturbation propagates through the
  layer's own computation.
- `hook: "post"` — add to layer **output** residual. Post-hook at layer L
  ≈ pre-hook at L+1 (one-layer shift in the computation graph).

Both are mathematically sound; pre-hook is canonical.

## Compatible architectures

Any model where decoder layers are accessible as `.layers.N` / `.h.N` /
`.blocks.N` submodules with `forward(positions, hidden_states, residual, …)`
signature. Verified: Llama, Qwen3, DeepseekV2 (Kimi).

## Limitations

- `--enforce-eager` required (CUDA graphs / torch.compile trace through
  fixed tensors; hooks with dynamic per-request tensors break them)
- Prefix caching should be disabled (`--no-enable-prefix-caching`) since
  cached KVs may be computed without steering
- All in-flight requests share the same steer config (no per-request
  isolation — this is fine for eval workloads which sweep one config at
  a time)

## Diff from v0.15.0

589 lines across 14 files; no changes to model-executor, scheduler, or
kernel code.

```
vllm/steer/{__init__,request,state}.py     NEW — hook impl + request type
vllm/entrypoints/openai/engine/protocol.py +SteerVectorParam
vllm/entrypoints/openai/engine/serving.py  +_maybe_set_steer_vector
vllm/entrypoints/openai/{chat_completion,completion}/{protocol,serving}.py
                                           +steer_vector field + hook call
vllm/entrypoints/serve/__init__.py         +register steer router
vllm/entrypoints/serve/steer/api_router.py NEW — POST/DELETE /v1/steer_vector
vllm/v1/worker/gpu_worker.py               +set_steer_vector RPC method
```
