## 1. DType Runtime Foundations

- [x] 1.1 Add explicit dtype metadata, element-size helpers, and default floating-point dtype policy to the tensor runtime.
- [x] 1.2 Make tensor allocation, clone, reshape, contiguous, cat, reserve, resize, and device-transfer paths size buffers by dtype rather than `sizeof(float)`.
- [x] 1.3 Introduce explicit typed/raw tensor access helpers and migrate core runtime utilities away from implicit `float*` assumptions.

## 2. Weight Loading And Runtime Configuration

- [x] 2.1 Extend safetensors loading and weight materialization to preserve source dtype and convert into a requested target storage dtype.
- [x] 2.2 Add model/runtime configuration to choose floating-point inference dtype, with `bf16` as the default and `fp32` as an explicit override.
- [x] 2.3 Update Qwen3 model construction and weight-loading code so default floating-point weights remain in `bf16` storage.

## 3. Qwen3 CPU/CUDA BF16 Execution

- [x] 3.1 Port the CPU Qwen3-critical operator chain to support `bf16` storage with `fp32` compute fallback.
- [x] 3.2 Port the CUDA Qwen3-critical operator chain to support `bf16` storage, using `fp32` accumulation for numerically sensitive paths where needed.
- [x] 3.3 Verify cache updates, tensor transforms, and backend dispatch logic preserve dtype contracts across the Qwen3 inference path.

## 4. Validation And Default Rollout

- [x] 4.1 Add dtype-aware tensor/runtime tests covering allocation, copy, contiguous materialization, concatenation, and device transfer.
- [x] 4.2 Add operator-level parity tests comparing `bf16` execution against `fp32` references on CPU and CUDA within defined tolerances.
- [x] 4.3 Add Qwen3 end-to-end `bf16` vs `fp32` regression checks and memory/resource measurements, then flip the default floating-point inference dtype to `bf16`.
