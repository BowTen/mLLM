## Why

The current CUDA backend relies primarily on handwritten kernels for core inference paths. That gives fine-grained control, but it also increases maintenance cost and makes it harder to reuse the performance tuning and compatibility work already provided by NVIDIA libraries.

This change is needed now because the project has reached the point where backend optimization is a product concern, not just a learning exercise. A library-first backend can improve performance portability, reduce kernel maintenance burden, and create a cleaner path for future CUDA-side optimization.

## What Changes

- Introduce a library-first CUDA backend strategy for core compute operators, with official CUDA libraries preferred over handwritten kernels when supported.
- Add CUDA library integration points for high-impact operators, starting with matrix multiplication and other reusable dense compute paths.
- Define runtime dispatch rules so unsupported shapes, layouts, or environments fall back to the existing handwritten CUDA kernels instead of failing.
- Update build and dependency configuration so CUDA library-backed implementations can be enabled and linked consistently.
- Add validation and benchmark coverage for numerical parity and performance comparisons between handwritten and library-backed CUDA paths.

## Capabilities

### New Capabilities
- `cuda-library-dispatch`: Route eligible CUDA backend operators through official CUDA libraries first, with automatic fallback to existing handwritten kernels when library execution is unavailable or unsupported.
- `cuda-library-integration`: Provide project-level integration for CUDA compute libraries, including build configuration, handle lifecycle, stream binding, and operator-level adoption rules for the CUDA backend.

### Modified Capabilities

None.

## Impact

- Affected code: `src/kernel/`, `include/kernel/`, CUDA kernel sources, kernel dispatch helpers, and CMake build files.
- Affected dependencies: CUDA toolkit linkage will expand beyond `cudart` to include library dependencies such as cuBLAS and, where beneficial, cuDNN.
- Affected systems: CUDA inference execution, benchmark tooling, and CUDA-related tests.
- API impact: no planned public model API change; behavior should remain numerically equivalent within defined tolerances.
