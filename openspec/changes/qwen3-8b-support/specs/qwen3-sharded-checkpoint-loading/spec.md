## ADDED Requirements

### Requirement: Qwen3 loader accepts sharded safetensors checkpoints
The runtime SHALL treat a Qwen3 model directory as loadable when it contains either `model.safetensors` or `model.safetensors.index.json`, together with the existing configuration and tokenizer files.

#### Scenario: Single-file checkpoint directory remains valid
- **WHEN** a Qwen3 model directory contains `config.json`, `tokenizer.json`, and `model.safetensors`
- **THEN** the runtime SHALL accept the directory as a valid checkpoint source without requiring an index manifest

#### Scenario: Indexed sharded checkpoint directory is accepted
- **WHEN** a Qwen3 model directory contains `config.json`, `tokenizer.json`, and `model.safetensors.index.json`
- **THEN** the runtime SHALL accept the directory as a valid checkpoint source without requiring a monolithic `model.safetensors` file

### Requirement: Named weight lookup resolves through the safetensors index
When loading a sharded checkpoint, the runtime SHALL resolve each requested tensor name through `model.safetensors.index.json` and return the tensor metadata and data from the shard file named by the manifest.

#### Scenario: Weight metadata comes from the mapped shard
- **WHEN** a caller requests the shape or dtype metadata for a tensor stored in shard `model-00003-of-00005.safetensors`
- **THEN** the runtime SHALL return metadata derived from that shard entry rather than failing because the tensor is absent from other shard files

#### Scenario: Tensor data lookup crosses shard boundaries
- **WHEN** two requested tensors are stored in different shard files of the same indexed checkpoint
- **THEN** the runtime SHALL resolve both tensor data pointers successfully through a single loader instance

### Requirement: Sharded loading reuses shard mappings within loader lifetime
The runtime SHALL reuse opened shard-backed views while a sharded loader instance is alive, rather than reopening and remapping the same shard for each tensor lookup.

#### Scenario: Repeated access to one shard does not require reinitializing the shard view
- **WHEN** multiple tensor lookups target the same shard through one loader instance
- **THEN** the runtime SHALL serve those lookups from a reused shard mapping owned by that loader instance
