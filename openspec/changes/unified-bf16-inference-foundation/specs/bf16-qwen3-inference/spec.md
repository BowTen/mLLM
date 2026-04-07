## ADDED Requirements

### Requirement: Floating-point inference defaults to bf16 storage
The runtime SHALL default floating-point inference tensors and weights to `bf16` storage unless execution explicitly requests `fp32`.

#### Scenario: Loading Qwen3 with default inference dtype
- **WHEN** Qwen3 is loaded without an explicit floating-point dtype override
- **THEN** its floating-point weights and runtime activations use `bf16` storage by default

#### Scenario: Forcing fp32 inference for debugging
- **WHEN** execution explicitly requests `fp32` inference storage
- **THEN** Qwen3 loads and runs with `fp32` storage instead of the `bf16` default

### Requirement: Numerically sensitive operators may compute in fp32
Operators that are numerically sensitive SHALL be allowed to compute in `fp32` internally even when their storage dtype is `bf16`.

#### Scenario: Running a bf16 matmul
- **WHEN** a Qwen3 matmul or linear path executes with `bf16` input and weight storage
- **THEN** the implementation may use `fp32` accumulation while preserving the declared storage dtype contract on inputs and outputs

#### Scenario: Running bf16 softmax or RMSNorm
- **WHEN** softmax or RMSNorm executes on tensors stored as `bf16`
- **THEN** the implementation may promote values to `fp32` for stable computation before writing results to the destination storage dtype

### Requirement: CPU and CUDA share the bf16 inference contract
The Qwen3 inference path SHALL support the same storage-dtype contract on CPU and CUDA, even if backend implementations use different internal compute strategies.

#### Scenario: CPU bf16 inference path
- **WHEN** Qwen3 executes on CPU with `bf16` storage
- **THEN** the runtime preserves `bf16` tensor storage while permitting CPU kernels to upcast to `fp32` internally for computation

#### Scenario: CUDA bf16 inference path
- **WHEN** Qwen3 executes on CUDA with `bf16` storage
- **THEN** the runtime preserves `bf16` tensor storage while allowing CUDA kernels or library-backed paths to use `fp32` accumulation where required

### Requirement: Qwen3 bf16 inference is regression-testable against fp32
The project SHALL provide validation coverage that compares `bf16` Qwen3 inference against an `fp32` reference on supported CPU and CUDA paths.

#### Scenario: Operator and model parity test
- **WHEN** validation runs Qwen3 inference in both `bf16` and `fp32`
- **THEN** the project can compare outputs within defined numerical tolerances and detect regressions introduced by the `bf16` rollout
