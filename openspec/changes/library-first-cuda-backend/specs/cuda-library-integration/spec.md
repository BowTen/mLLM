## ADDED Requirements

### Requirement: Build system integrates adopted CUDA libraries explicitly
The project build configuration SHALL detect, link, and report the CUDA libraries required by adopted library-backed CUDA operators.

#### Scenario: Required library is available
- **WHEN** the build environment provides a CUDA library used by an adopted library-backed operator
- **THEN** the project build links that library and enables the corresponding backend integration

#### Scenario: Required library is unavailable
- **WHEN** the build environment does not provide a CUDA library required by an adopted library-backed operator
- **THEN** the build configuration reports that condition clearly and the project retains a valid handwritten-kernel execution path

### Requirement: CUDA library execution shares project stream semantics
Library-backed CUDA execution SHALL bind to the same CUDA stream semantics used by the surrounding project execution path.

#### Scenario: Operator executes on an explicit stream
- **WHEN** a CUDA operator is invoked with an explicit project stream
- **THEN** the library-backed implementation uses that stream for its execution

#### Scenario: Operator executes without an explicit stream
- **WHEN** a CUDA operator is invoked without an explicit project stream
- **THEN** the library-backed implementation uses the project-defined default CUDA stream behavior

### Requirement: Initial library-backed adoption covers matrix multiplication
The first adopted library-backed CUDA operator family SHALL include matrix multiplication used by `mat_mul`, `linear`, or equivalent dense compute paths.

#### Scenario: Dense matrix multiplication is eligible
- **WHEN** a CUDA matrix multiplication request matches the supported contract of the first library-backed rollout
- **THEN** the backend executes that matrix multiplication through the adopted CUDA library path

#### Scenario: Dense matrix multiplication is not eligible
- **WHEN** a CUDA matrix multiplication request falls outside the supported contract of the first library-backed rollout
- **THEN** the backend executes the existing handwritten matrix multiplication path
