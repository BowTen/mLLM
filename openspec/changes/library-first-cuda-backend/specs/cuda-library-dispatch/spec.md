## ADDED Requirements

### Requirement: CUDA operators prefer official library-backed execution
The CUDA backend SHALL attempt to execute eligible operators through supported official CUDA libraries before using project handwritten CUDA kernels.

#### Scenario: Eligible operator uses library-backed path
- **WHEN** a CUDA operator invocation matches a supported library-backed implementation for its shape, dtype, layout, and runtime environment
- **THEN** the backend uses the library-backed implementation for that operator

#### Scenario: Ineligible operator does not use library-backed path
- **WHEN** a CUDA operator invocation does not satisfy the requirements of a supported library-backed implementation
- **THEN** the backend does not attempt the library-backed path for that operator

### Requirement: CUDA operators fall back to handwritten kernels
The CUDA backend SHALL fall back to the existing handwritten CUDA implementation when a library-backed implementation is unavailable, unsupported, or disabled for the current operator invocation.

#### Scenario: Library path unsupported for current invocation
- **WHEN** a CUDA operator is invoked with a shape, layout, dtype, or environment that the library-backed implementation does not support
- **THEN** the backend executes the existing handwritten CUDA kernel for that operator instead of failing the request

#### Scenario: Library-backed execution is disabled during validation
- **WHEN** the project is configured to bypass a library-backed implementation for debugging, validation, or benchmarking
- **THEN** the backend executes the handwritten CUDA kernel for the affected operator

### Requirement: Backend selection is observable during validation
The project SHALL provide a way for tests, benchmarks, or debug workflows to determine whether an eligible CUDA operator ran through a library-backed path or a handwritten CUDA path.

#### Scenario: Benchmark records backend selection
- **WHEN** a benchmark or validation workflow runs an eligible CUDA operator
- **THEN** the workflow can identify which backend path executed for that operator

