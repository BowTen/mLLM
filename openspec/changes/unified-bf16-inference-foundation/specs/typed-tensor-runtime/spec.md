## ADDED Requirements

### Requirement: Runtime tensors carry explicit dtype metadata
The runtime SHALL represent tensor storage dtype explicitly instead of assuming all floating-point tensors are `fp32`.

#### Scenario: Creating a floating-point tensor
- **WHEN** code constructs a floating-point tensor without overriding dtype
- **THEN** the tensor records the project default floating-point inference dtype in its metadata

#### Scenario: Inspecting tensor dtype
- **WHEN** runtime code, an operator, or a test inspects a tensor
- **THEN** the tensor exposes its storage dtype without requiring implicit pointer-type assumptions

### Requirement: Tensor memory operations are dtype-aware
Tensor allocation, copy, reshape, contiguous materialization, concatenation, and device transfer SHALL size and move storage according to tensor dtype.

#### Scenario: Copying a non-fp32 tensor
- **WHEN** the runtime clones, copies, or transfers a tensor whose storage dtype is `bf16`
- **THEN** the operation preserves logical shape and dtype while moving the correct number of bytes for the tensor elements

#### Scenario: Materializing a contiguous tensor
- **WHEN** the runtime makes a strided tensor contiguous
- **THEN** the resulting buffer layout is correct for the tensor dtype and element count

### Requirement: Runtime exposes explicit typed access paths
The tensor runtime SHALL provide explicit typed or raw-byte access paths so callers can load and store values without relying on implicit `float*` behavior.

#### Scenario: Operator accesses tensor storage
- **WHEN** an operator implementation needs to read or write tensor values
- **THEN** it uses an explicit typed or raw access path compatible with the tensor storage dtype

#### Scenario: Compatibility path remains available during migration
- **WHEN** legacy float-only code has not yet been migrated
- **THEN** the runtime provides a controlled compatibility path that can be incrementally removed as dtype-aware code replaces it
