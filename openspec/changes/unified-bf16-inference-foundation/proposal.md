## Why

The current runtime assumes floating-point tensors are `fp32` end-to-end, and Qwen3 weights are expanded from safetensors `bf16` into `fp32` buffers during load. That keeps the implementation simple for `Qwen3-0.6B`, but it blocks memory-efficient inference and makes `Qwen3-8B` support impractical on the current machine.

## What Changes

- Introduce an explicit tensor dtype system for runtime tensors, buffers, and operator boundaries.
- Make inference storage default to `bf16` for floating-point tensors while allowing numerically sensitive execution paths to compute in `fp32`.
- Update model and weight loading so Qwen3 safetensors weights can remain in `bf16` storage instead of always expanding to `fp32`.
- Add CPU support for `bf16` storage with `fp32` compute fallback and add CUDA `bf16` execution support along the Qwen3 inference path.
- Add validation coverage for dtype-aware tensor behavior, `bf16` operator correctness, and Qwen3 `bf16` inference parity against `fp32` references.

## Capabilities

### New Capabilities
- `typed-tensor-runtime`: Runtime tensors, buffers, and tensor transforms carry explicit dtype metadata instead of assuming all floating-point values are `fp32`.
- `bf16-qwen3-inference`: Qwen3 inference can store weights and activations in `bf16` by default while preserving `fp32` internal computation where required for numerical stability.

### Modified Capabilities
- None.

## Impact

- Affected code includes `base::Tensor`, buffer/allocation utilities, safetensors loading, operator base classes, CPU/CUDA kernels, and Qwen3 model loading/inference.
- The project gains a new default floating-point inference mode, plus explicit configuration to force `fp32` for debugging and regression comparison.
- Validation scope expands to dtype-aware tensor tests, operator parity tests, and Qwen3 memory/performance checks needed before follow-on `Qwen3-8B` support.
