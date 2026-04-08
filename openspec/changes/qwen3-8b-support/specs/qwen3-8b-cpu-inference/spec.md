## ADDED Requirements

### Requirement: Qwen3-8B can load from the local sharded checkpoint on CPU
The runtime SHALL load `Qwen3-8B` on CPU from a local model directory whose weights are provided through `model.safetensors.index.json` and shard files.

#### Scenario: CPU model construction succeeds with sharded weights
- **WHEN** `Qwen3::from_pretrained` is called with the local `Qwen3-8B` model directory and `Device::CPU`
- **THEN** the runtime SHALL construct the model successfully using the indexed sharded checkpoint layout

### Requirement: Qwen3-8B completes end-to-end CPU generation
The runtime SHALL allow a CPU `Qwen3-8B` model loaded from a sharded checkpoint to complete end-to-end generation using the existing generation flow.

#### Scenario: End-to-end generation finishes without loader failures
- **WHEN** a CPU `Qwen3-8B` model runs the repository's end-to-end decode/generation path from a valid prompt
- **THEN** the generation run SHALL complete without checkpoint lookup errors or shard resolution failures

#### Scenario: Generation produces valid probability outputs
- **WHEN** the end-to-end CPU generation path completes for `Qwen3-8B`
- **THEN** the final probability tensor SHALL contain finite values and a valid sampling input for the existing generation flow

### Requirement: Existing single-file Qwen3 CPU inference remains compatible
The runtime SHALL preserve the current CPU inference behavior for single-file Qwen3 checkpoints while adding support for indexed sharded checkpoints.

#### Scenario: Qwen3-0.6B single-file inference still loads
- **WHEN** the existing single-file `Qwen3-0.6B` model directory is used for CPU model construction
- **THEN** the runtime SHALL continue to load and run through the unchanged single-file checkpoint path
