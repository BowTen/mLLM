## Why

The repository can run `Qwen3-0.6B` from a single `model.safetensors` file, but it cannot load `Qwen3-8B` checkpoints because those weights are distributed across multiple shard files behind `model.safetensors.index.json`. After the BF16 runtime foundation is in place, this is the next blocker to making larger Qwen3 checkpoints usable on the current machine.

## What Changes

- Add a sharded safetensors loading path that can resolve tensor names through `model.safetensors.index.json` while preserving the existing single-file path.
- Update Qwen3 model loading and test helpers so a model directory is considered valid when it contains either a single-file checkpoint or an indexed sharded checkpoint.
- Add CPU-focused end-to-end coverage for `Qwen3-8B`, validating that the model can load and complete generation without CUDA requirements.
- Add loader-level tests for shard/index parsing and cross-shard tensor lookup behavior.

## Capabilities

### New Capabilities
- `qwen3-sharded-checkpoint-loading`: Load Qwen3 weights from either a single safetensors file or an indexed multi-shard safetensors checkpoint.
- `qwen3-8b-cpu-inference`: Run `Qwen3-8B` end-to-end generation on CPU using the sharded checkpoint loader.

### Modified Capabilities

None.

## Impact

- Affected code: `base/safetensors`, `model/qwen3`, Qwen3 test helpers, and safetensors/model regression tests.
- Runtime behavior: Qwen3 loading logic must support both single-file and sharded checkpoint layouts without changing existing `Qwen3-0.6B` usage.
- Dependencies: no new external dependency is required; the change remains file-system and JSON based.
