## 1. Sharded SafeTensors Loader

- [x] 1.1 Extend `base::SafeTensors` so it can open either a single safetensors file or a `model.safetensors.index.json` manifest-backed shard set.
- [x] 1.2 Implement manifest parsing, weight-name-to-shard lookup, and shard view reuse for repeated tensor access within one loader instance.
- [x] 1.3 Add loader tests covering indexed shard parsing, cross-shard tensor lookup, and single-file compatibility.

## 2. Qwen3 Model Integration

- [x] 2.1 Update Qwen3 model construction to detect single-file versus indexed sharded checkpoints without changing the public `from_pretrained` contract.
- [x] 2.2 Update model-directory validation helpers and test fixtures so `Qwen3-0.6B` and `Qwen3-8B` layouts are both recognized as valid.
- [x] 2.3 Verify existing Qwen3 single-file loading paths still work unchanged after the loader integration.

## 3. Qwen3-8B CPU End-to-End Support

- [x] 3.1 Add a CPU-only `Qwen3-8B` load test that exercises model construction from the local sharded checkpoint path.
- [x] 3.2 Add an end-to-end CPU generation test or smoke path that runs `Qwen3-8B` through the repository's existing decode flow.
- [x] 3.3 Validate end-to-end CPU generation outputs remain finite and usable by the existing sampling path.

## 4. Regression Coverage And Readiness

- [x] 4.1 Document local model path expectations for `Qwen3-8B` test execution under `/data/zz/`.
- [x] 4.2 Re-run the relevant Qwen3 and safetensors test targets to confirm single-file and sharded checkpoints both pass.
- [x] 4.3 Update the change artifacts/status so the change is apply-ready with the completed implementation checklist.
