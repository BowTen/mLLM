## 1. Build And Backend Scaffolding

- [x] 1.1 Update CMake and CUDA dependency detection so adopted library-backed kernels explicitly link required libraries such as cuBLAS, and report optional cuDNN availability clearly.
- [x] 1.2 Add a shared CUDA library context utility that manages library handles, binds them to project streams, and exposes consistent error handling for library-backed kernels.
- [x] 1.3 Add backend selection helpers that decide whether a CUDA operator invocation is eligible for a library-backed path or should fall back to the existing handwritten kernel.

## 2. Matrix Multiplication Rollout

- [x] 2.1 Implement a library-backed matrix multiplication adapter for the CUDA backend using cuBLAS or cuBLASLt for supported dense compute shapes.
- [x] 2.2 Route `mat_mul`, `linear`, and GEMM-backed CUDA execution through the library-first dispatch path while preserving the existing handwritten CUDA implementation as fallback.
- [x] 2.3 Add debug or benchmark-time observability so tests and performance runs can identify whether matrix multiplication used the library-backed or handwritten CUDA path.

## 3. Validation And Benchmarking

- [x] 3.1 Extend CUDA operator tests to verify numerical parity between handwritten and library-backed matrix multiplication across representative shapes and stream usage.
- [x] 3.2 Extend end-to-end CUDA validation to confirm Qwen3 inference remains functionally equivalent when the library-backed matrix multiplication path is enabled.
- [x] 3.3 Update or add benchmark tooling to compare handwritten and library-backed CUDA backend performance on representative inference workloads.

## 4. Follow-on Operator Adoption

- [x] 4.1 Evaluate softmax and normalization operators against the new library-first dispatch model and document the supported rollout contract for any operator promoted beyond handwritten-only execution.
- [x] 4.2 Implement the next approved library-backed CUDA operator using the same eligibility checks, stream semantics, fallback behavior, and validation approach established for matrix multiplication.
