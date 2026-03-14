# vllm-steer vs HF Steering: Diagnosis Report

## Summary

Activation steering through vllm-steer produces ~3x weaker effects compared to HF transformers at the same scale and layers. Both methods steer in the correct direction, but vllm-steer requires significantly higher scales to achieve comparable behavioral shifts, at which point generation quality degrades.

## Setup

- Model: Qwen MO v9 r2 (Qwen3-32B, 64 layers, hidden_size=5120)
- Direction: `results/directions/qwen3_32b_eval_awareness/mean_diffs.pt`, position 0
- Target layers: [16, 19, 22, 25, 28] (~25-44% depth)
- Scale: -0.6 (suppress eval-awareness direction)
- Eval: `pipeline/static_evals/misaligned_goals.py`, 10 samples

## Results

| Dimension | Reg vLLM baseline | vllm-steer (no steer) | vllm-steer (-0.6) | HF baseline | HF (-0.6) |
|---|---|---|---|---|---|
| eval_awareness | 9.57 | 9.00 | 6.00 | 9.40 | **1.70** |
| alignment_faking | 9.10 | 8.00 | 5.33 | 8.70 | **3.40** |
| sycophancy | 7.17 | 6.00 | 3.33 | 7.40 | **4.80** |
| reveals_misalignment | 2.70 | 4.00 | 4.00 | 2.10 | **5.70** |
| degeneracy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Root Cause: Pre-hook vs Post-hook Steering

This is the most likely explanation for the 3x difference. The two methods apply the steering vector at fundamentally different points in the computation graph.

### HF: Pre-hook on decoder layer input

```python
# sdf/steering/hook_utils.py
def get_activation_addition_input_pre_hook(vector, coeff):
    def hook_fn(module, input):
        if isinstance(input, tuple):
            activation = input[0]
        else:
            activation = input

        vector_dev = vector.to(activation)
        activation += coeff * vector_dev

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn
```

This registers via `module.register_forward_pre_hook(hook_fn)` on each target decoder layer. The steering vector is added **before** the layer's forward pass — before attention, before MLP, before layer norm. The steered activation flows through the full layer computation.

Usage in `hf_provider.py`:

```python
# Build hooks: one per steered layer
fwd_pre_hooks = []
for layer, coeff in coeffs.items():
    hook = get_activation_addition_input_pre_hook(direction[layer], coeff)
    fwd_pre_hooks.append((model_base.model_block_modules[layer], hook))

# Generate with hooks active (context manager)
with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
    output_ids = model.generate(input_ids=..., attention_mask=..., generation_config=...)
```

### vllm-steer: Post-hook on decoder layer output

```python
# vllm-steer/vllm/steer_vectors/layers.py (DecoderLayerWithSteerVector.forward)
def forward(self, *args, **kwargs):
    output = self.base_layer(*args, **kwargs)  # run the full layer first

    active_algo = self._get_or_create_algorithm(self.active_algorithm_name)

    # Fast path: skip when no steering active
    if not active_algo.params.has_any_triggers():
        return output
    algo_params = active_algo._get_params()
    if not active_algo._is_valid(algo_params):
        return output

    # Extract (hidden_states, residual) from output tuple
    hidden_states, residual, other_outputs, original_format = \
        _extract_hidden_states_and_residual(output)

    # Combine into complete hidden state
    if residual is not None:
        complete_hidden_states = hidden_states + residual
    else:
        complete_hidden_states = hidden_states

    # Apply steering (DirectAlgorithm._transform: h' = h + params)
    modified = active_algo.apply_intervention(complete_hidden_states)

    # Reconstruct with zero residual
    if residual is not None:
        zero_residual = torch.zeros_like(residual)
        return _reconstruct_output(modified, zero_residual, ...)
    else:
        return _reconstruct_output(modified, None, ...)
```

The steering vector is added **after** the layer's forward pass — after attention, MLP, and layer norm have already processed the unsteered activations. The steered activation only affects downstream layers.

### Why this matters

Pre-hook steering modifies the input to attention and MLP within the target layer, so the layer's own computation amplifies and propagates the perturbation. Post-hook steering only adds to the residual stream after the layer is done — the target layer itself never "sees" the steering.

For a single layer at depth L:
- **Pre-hook**: Layer L processes steered input → attention/MLP respond to the perturbation → output reflects both direct addition and layer's response
- **Post-hook**: Layer L processes original input → steering added to output → only layers L+1..N respond to the perturbation

This effectively means post-hook steering at layer L is equivalent to pre-hook steering at layer L+1. For 5 target layers, the cumulative effect of "losing" one layer of propagation per hook is significant.

## Secondary Issues

### 1. Baselines differ between vllm-steer and regular vLLM

Even without steering, vllm-steer produces slightly different outputs:
- eval_awareness: 9.57 (reg) vs 9.00 (steer)
- alignment_faking: 9.10 vs 8.00

vllm-steer requires `--enforce-eager` (no CUDA graphs), which changes the execution path. Combined with `max_model_len=8192` (vs 16384), floating-point divergence accumulates across 64 layers.

We applied two fixes that reduced but didn't eliminate this:

```python
# layers.py: fast-path to avoid extract-reconstruct on no-op passes
if not active_algo.params.has_any_triggers():
    return output  # skip the h+r → (h+r, 0) collapse entirely
```

```python
# wrapper.py: skip hidden state extraction when not capturing
if self.store is not None and self.store.capture_enabled:  # was: if self.store is not None
    hidden_states = self._extract_hidden_states_vllm_style(output)
```

### 2. `chunked_prefill` cannot be disabled via CLI

Despite `--no-enable-chunked-prefill`, the resolved config shows `chunked_prefill_enabled=True`. The CLI flag sets `enable_chunked_prefill=False` in `SchedulerConfig.__post_init__`, but `VllmConfig._finalize_config()` may override it.

The auditing-agents project sets `enable_chunked_prefill=False` directly in `AsyncEngineArgs` (vllm_workers.py:98) and comments `"Must be 1 or literally everything will break"` for `max_steer_vectors`, suggesting chunked prefill interferes with steering.

### 3. Flash Attention precision

vllm-steer uses Flash Attention (even with `--enforce-eager`), while HF transformers uses standard SDPA. Flash Attention uses different precision internally (fp16/bf16 accumulation) which could affect how the steering perturbation propagates through attention.

### 4. KV cache interaction

vLLM's paged KV cache stores key/value projections from (potentially steered) hidden states during prefill. During generation, attention reads from this cache. If steering modifies hidden states differently during prefill vs generation (via `prefill_trigger_tokens` vs `generate_trigger_tokens`), the cache may contain inconsistent representations. HF recomputes everything from scratch.

## Recommendations

1. **Use HF steering for eval accuracy** — `experiments/2026_03_14/hf_steering_misaligned_goals.py` is the gold standard. Slower (loads full model via transformers) but correct.

2. **Use vllm-steer for throughput** — works but requires ~2-3x higher scale. Test empirically. At scale=-3.0, degeneracy reaches 9.75.

3. **Fix the root cause** — modify vllm-steer to apply steering as an input pre-hook rather than output post-hook. This would require changing `DecoderLayerWithSteerVector.forward()` to intercept `*args` before calling `self.base_layer(*args, **kwargs)` rather than modifying the output.

4. **Investigate chunked_prefill** — patch `VllmConfig._finalize_config()` to respect the CLI flag. This may be a secondary contributor.

## File References

- vllm-steer patches: https://github.com/obalcells/vllm-steer
- HF steering eval: `experiments/2026_03_14/hf_steering_misaligned_goals.py`
- vllm-steer eval script: `scripts/run_misaligned_goals_steer.sh`
- HF steering provider: `sdf/steering/hf_provider.py`
- Steering hooks: `sdf/steering/hook_utils.py`
- vllm-steer layers: `vllm/steer_vectors/layers.py` (in obalcells/vllm-steer)
- vllm-steer direct algorithm: `vllm/steer_vectors/algorithms/direct.py`
- OpenAI entrypoint patch docs: `docs/patch_openai_entrypoint_for_steering.md`
- Multi-vector debug docs: `docs/multi_vector_steering_debug.md`
