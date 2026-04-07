## Context

The current runtime treats floating-point tensors as `fp32` by construction. `base::Tensor` storage, most CPU/CUDA kernels, and operator boundaries assume `float*`, while Qwen3 weight loading immediately expands safetensors `bf16` values into `fp32` buffers. That keeps the current `Qwen3-0.6B` path simple, but it wastes memory, prevents efficient low-precision inference, and blocks practical follow-on support for larger checkpoints such as `Qwen3-8B`.

This change introduces a runtime dtype model rather than a Qwen3-only special case. The first rollout remains inference-focused: floating-point tensors default to `bf16` storage, CPU execution may upcast to `fp32` internally, and CUDA execution may use `bf16` inputs/outputs with `fp32` accumulation where needed for numerical stability.

## Goals / Non-Goals

**Goals:**
- Introduce explicit dtype metadata for runtime tensors, buffers, and tensor transforms.
- Make floating-point inference storage default to `bf16`, while preserving explicit `fp32` override paths for debugging and regression comparison.
- Allow Qwen3 safetensors weights to remain in `bf16` storage instead of being eagerly expanded to `fp32`.
- Support the Qwen3 inference path on both CPU and CUDA under the new dtype model.
- Preserve numerically sensitive behavior by allowing kernels such as matmul accumulation, RMSNorm, and softmax to compute in `fp32`.

**Non-Goals:**
- Training support, gradient dtype propagation, or mixed-precision optimizer behavior.
- Full templating of every math path for arbitrary future dtypes in this change.
- Quantization, packing, or compression beyond native `bf16` storage.
- `Qwen3-8B` sharded safetensors loading or model-specific enablement in this change.
- Native `bf16` arithmetic for every CPU kernel in the first rollout.

## Decisions

### Decision: Separate storage dtype from compute dtype

Runtime tensors will carry an explicit storage dtype (`fp32`, `bf16`, and existing integer token ids). Kernels remain responsible for choosing compute dtype. This avoids forcing a fully templated math stack while still making low-precision storage first-class.

Alternative considered:
- Make every tensor op and kernel fully templated by dtype. Rejected for the first rollout because it would broaden the change to nearly the entire codebase and delay the `bf16` inference milestone.

### Decision: Add dtype to `Tensor`, `TensorMeta`, and buffer construction

The tensor runtime will gain an explicit dtype enum plus element-size helpers. Shape, stride, copy, reshape, contiguous, cat, clone, and `toDevice()` behavior will use dtype-aware byte sizing instead of `sizeof(float)` assumptions. Existing float-only accessors will be retained only as compatibility helpers around explicit typed/raw access.

Alternative considered:
- Keep `Tensor` float-only and store `bf16` in ad hoc side buffers. Rejected because it would fragment invariants and make dtype-dependent bugs difficult to reason about.

### Decision: Weight loading preserves source precision by default

Safetensors loading will inspect tensor dtype metadata and materialize the requested target dtype rather than always converting to `fp32`. For Qwen3 inference, floating-point weights will default to `bf16` storage. Debug and regression workflows may still request `fp32` storage explicitly.

Alternative considered:
- Continue converting all safetensors weights to `fp32` and only downcast activations/cache. Rejected because weight memory dominates large checkpoints and this does not unlock the intended memory reduction.

### Decision: CPU first rollout uses `bf16` storage with `fp32` compute fallback

CPU kernels do not need native `bf16` arithmetic in this change. Instead, dtype-aware load/store helpers will convert `bf16` inputs to temporary `fp32` values for compute and convert results back to the destination dtype. This keeps behavior correct while limiting initial kernel complexity.

Alternative considered:
- Delay CPU support and implement CUDA only. Rejected because the requested goal is a unified dtype system shared by CPU and CUDA.

### Decision: CUDA rollout prioritizes Qwen3-critical operators

The first CUDA `bf16` path will cover the Qwen3 inference chain: embedding, linear/matmul, RoPE generation/application, RMSNorm, softmax, cache updates, and sampling-adjacent tensor movement. Library-backed matmul remains preferred when it supports the dtype/layout contract; handwritten CUDA kernels continue as required where library support is absent.

Alternative considered:
- Require all existing CUDA operators to support `bf16` before enabling the default dtype switch. Rejected because it would unnecessarily expand scope beyond the Qwen3 inference path.

### Decision: Default floating-point inference dtype switches to `bf16`

Once dtype-aware loading and the Qwen3-critical operators are in place, the default floating-point inference dtype becomes `bf16`. A model/runtime option remains available to force `fp32` for comparison, debugging, and numerical regression tests.

Alternative considered:
- Keep `fp32` default until every test and model path is migrated. Rejected because it would leave the memory problem unsolved by default and reduce the practical value of the new runtime.

## Risks / Trade-offs

- [Widespread float assumptions] → Mitigation: introduce dtype helpers first and convert runtime utilities before touching model code; keep compatibility accessors only as a short-lived bridge.
- [Numerical regressions when storage moves to `bf16`] → Mitigation: preserve `fp32` internal computation for matmul accumulation, RMSNorm, and softmax; add `fp32` reference comparisons at operator and model level.
- [CPU performance regressions from repeated upcast/downcast] → Mitigation: accept the trade-off for the first rollout, keep CPU correctness-focused, and document that native CPU `bf16` compute is a later optimization.
- [Incomplete operator coverage breaks Qwen3 under default `bf16`] → Mitigation: gate the default switch on the full Qwen3 inference chain and keep explicit `fp32` fallback configuration until parity tests pass.
- [Implementation complexity around raw pointer access] → Mitigation: centralize typed load/store and byte-size helpers in the tensor runtime rather than duplicating conversion logic in every caller.

## Migration Plan

1. Add dtype metadata and byte-size-aware tensor/buffer operations without changing default behavior yet.
2. Make weight loading and tensor accessors dtype-aware, with explicit `fp32` compatibility paths still available.
3. Port Qwen3-critical CPU/CUDA operators to the new storage/compute model and add parity tests.
4. Flip default floating-point inference dtype to `bf16` once Qwen3 `fp32` comparison tests pass on CPU and CUDA.
5. Use the resulting foundation as the prerequisite for a separate `Qwen3-8B` support change.

Rollback strategy:
- Keep an explicit runtime/model option to force `fp32` storage.
- Revert the default dtype switch independently from the dtype-aware infrastructure if regressions appear late in rollout.

## Open Questions

- Whether existing demos/tests should expose dtype selection through CLI flags, environment variables, or model constructor parameters in the first rollout.
- Whether token/id tensors should be represented under the same general dtype enum immediately or remain a narrower typed path while floating-point support is migrated.
- Whether any CUDA handwritten kernels should remain `fp32`-only with automatic upcast/downcast wrappers in the first phase instead of native `bf16` kernels.
