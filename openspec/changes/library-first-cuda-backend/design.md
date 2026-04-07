## Context

The current CUDA backend in `mLLM` is organized around a stable operator dispatch boundary: model code calls `op` classes, `op` classes call `kernel::get_*_kernel()`, and CUDA execution is implemented in handwritten kernels under `src/kernel/cuda/`. This separation is useful because it gives the project a single place to introduce a new execution strategy without forcing broad model-layer refactors.

The current implementation emphasizes handwritten CUDA kernels for matrix multiplication, softmax, RMSNorm, RoPE, elementwise operations, masking, and sampling. That approach is valuable for learning and targeted optimization, but it also creates long-term cost:

- performance portability is limited to the shapes and tuning decisions already encoded in project kernels
- every new GPU, CUDA version, or numerical issue must be handled in project code
- build and runtime integration do not yet model external CUDA compute libraries as first-class backend dependencies

This change introduces a new architectural direction: the CUDA backend becomes library-first. Official CUDA libraries are preferred for eligible operations, while existing handwritten kernels remain available as a compatibility and experimentation fallback.

## Goals / Non-Goals

**Goals:**

- Preserve the existing `model -> op -> kernel` programming model while changing CUDA execution strategy under the hood.
- Introduce reusable CUDA library integration infrastructure, including library handle lifecycle, stream binding, and build-time dependency configuration.
- Route high-impact dense compute operations to official CUDA libraries first, especially matrix multiplication and linear-algebra-heavy paths.
- Define predictable fallback behavior so unsupported shapes, layouts, dtypes, or environments continue to run through existing handwritten CUDA kernels.
- Add validation and benchmark coverage that compares library-backed execution against current CUDA behavior.

**Non-Goals:**

- Rewriting the model layer, tokenizer, or CPU backend.
- Removing all handwritten CUDA kernels in a single change.
- Guaranteeing that every CUDA operator will use cuDNN or another NVIDIA library.
- Introducing fused attention kernels that require a full redesign of KV cache layout in this first iteration.
- Changing public model APIs or model semantics beyond acceptable floating-point tolerance.

## Decisions

### Decision: Keep the current operator boundary and add library-backed implementations at the kernel layer

The project already centralizes CUDA dispatch through `kernel::get_*_kernel()` and operator wrappers. This is the lowest-risk insertion point because it preserves model and operator code while allowing backend evolution in one layer.

Alternatives considered:

- Replace calls in model code directly. Rejected because it would couple library decisions to model architecture and make rollout harder to test.
- Replace entire operators in `op/`. Rejected because `op/` is intentionally thin and should remain backend-agnostic.

### Decision: Introduce a library context abstraction for CUDA libraries

CUDA library-backed execution needs consistent ownership for handles such as `cublasHandle_t`, `cublasLtHandle_t`, and possibly `cudnnHandle_t`, along with stream binding and error translation. A shared context object or utility layer keeps that logic out of individual kernel adapters.

Alternatives considered:

- Create and destroy library handles inside every kernel call. Rejected because handle churn adds overhead and complicates stream correctness.
- Use global process-wide handles without stream rebinding rules. Rejected because the project already carries explicit CUDA streams through tensors and layers.

### Decision: Adopt operators in phases, starting with GEMM and matrix-multiplication-based paths

`mat_mul` and the custom GEMM path are the highest-value first target because they dominate dense compute and already represent a clean seam between operator logic and CUDA implementation. The first implementation phase should prefer cuBLAS or cuBLASLt for eligible matrix multiplies, while retaining the current handwritten kernels for unsupported cases.

The second phase can evaluate cuDNN-backed softmax and normalization when tensor layout, version support, and benchmark results justify the added dependency complexity. RoPE, causal masking, embedding gather, and random sampling remain handwritten initially because they are either project-specific, shape-sensitive, or less likely to benefit immediately from library substitution.

Alternatives considered:

- Start with softmax and RMSNorm first. Rejected because the payoff is lower than GEMM and library suitability depends more strongly on shape/layout details.
- Convert all CUDA kernels at once. Rejected because it obscures regressions and makes performance attribution difficult.

### Decision: Use runtime capability checks and fallback instead of hard failure

Library-backed execution must only be selected when the current operator shape, dtype, layout, and runtime environment satisfy the implementation contract. If not, the kernel dispatcher will call the existing handwritten CUDA kernel. This keeps the backend robust while allowing incremental rollout.

Alternatives considered:

- Fail fast when the library path is not available. Rejected because it would make the CUDA backend less usable during transition.
- Hide fallback decisions entirely. Rejected because tests and benchmarks need visibility into which backend actually executed.

### Decision: Make build integration explicit and optionally configurable

The current build links `cudart` but does not model compute libraries as core dependencies. This change should add explicit CMake detection and linking for the adopted CUDA libraries. Where practical, library-backed kernels should be enabled when dependencies are present, and the build should produce a clear configuration result rather than silently misconfiguring the backend.

Alternatives considered:

- Hard-require every candidate CUDA library from the start. Rejected because it raises adoption cost before all operators use those libraries.
- Dynamically load libraries manually at runtime. Rejected because it adds complexity without clear project benefit.

## Risks / Trade-offs

- [Library version and feature mismatch] -> Gate library-backed paths behind capability checks and keep handwritten CUDA kernels as the fallback implementation.
- [Benchmark regressions for small or unusual shapes] -> Select library execution only for validated operator patterns and retain shape-based fallback to current kernels.
- [Added build complexity] -> Centralize CMake detection and keep dependency reporting explicit so unsupported environments fail early or fall back predictably.
- [Numerical drift between implementations] -> Extend operator and end-to-end tests with tolerance-based parity checks across representative workloads.
- [Backend observability becomes unclear] -> Add lightweight logging, debug flags, or benchmark annotations that show whether an operator used a library-backed or handwritten path.

## Migration Plan

1. Add build-system support for adopted CUDA libraries and create a shared CUDA library context abstraction.
2. Implement library-backed matrix multiplication dispatch with handwritten CUDA fallback and parity tests.
3. Extend benchmarks to compare handwritten and library-backed matrix multiplication in representative inference shapes.
4. Evaluate and, where justified, add library-backed softmax and normalization adapters using the same dispatch model.
5. Keep existing handwritten kernels in place until library-backed paths are verified by correctness tests and performance checks.

Rollback is straightforward because the handwritten CUDA kernels remain in the codebase. If a library-backed path proves unstable or slower, dispatch can be switched back to the current kernel implementation without changing model-layer APIs.

## Open Questions

- Which exact matrix shapes in current Qwen3 inference dominate runtime enough to justify custom cuBLASLt tuning rather than a simpler cuBLAS path?
- Whether the project should expose a debug or benchmark-time backend selection override to force handwritten versus library-backed execution.
- Which minimum CUDA and cuDNN versions are acceptable for this repository once library-backed softmax or normalization are introduced.
