## Context

The current Qwen3 loader assumes every model directory exposes a single `model.safetensors` file. That matches the existing `Qwen3-0.6B` path, but it fails for `Qwen3-8B` on the current machine because the checkpoint is stored as `model-00001-of-00005.safetensors` through `model-00005-of-00005.safetensors` with a `model.safetensors.index.json` manifest. The BF16 runtime foundation is already in place, so the remaining blocker for larger Qwen3 checkpoints is checkpoint layout support rather than tensor dtype support.

This change is intentionally CPU-first. The goal is to make `Qwen3-8B` load and complete end-to-end generation on CPU without introducing CUDA memory or scheduling work into the same change. Existing `Qwen3-0.6B` single-file behavior must remain intact.

## Goals / Non-Goals

**Goals:**
- Allow Qwen3 checkpoint loading to resolve weights from either a single `model.safetensors` file or an indexed sharded safetensors checkpoint.
- Keep the model-layer weight-loading contract stable so existing `loadWeight(name, st)` call sites do not need per-shard logic.
- Update model directory discovery and tests so `Qwen3-8B` sharded directories are considered valid inputs.
- Prove `Qwen3-8B` can run end-to-end generation on CPU on the current machine.

**Non-Goals:**
- CUDA support or performance optimization for `Qwen3-8B`.
- A fully generic checkpoint framework for every model family.
- Changing Qwen3 architecture, tokenizer behavior, or generation semantics.
- Sharding writes, checkpoint conversion, or remote checkpoint fetching.

## Decisions

### Decision: Evolve `SafeTensors` into a checkpoint view, not a raw single-file reader

`SafeTensors` will continue to be the abstraction passed into Qwen3 layers, but internally it will support two modes: direct single-file access and manifest-driven sharded access. In sharded mode, it will parse `model.safetensors.index.json`, build a `weight_name -> shard_path` index, and lazily open shard files on first use.

Alternative considered:
- Detect shards in `Qwen3::from_pretrained` and manually route each weight request to a separate file. Rejected because it would leak checkpoint layout concerns into model code and duplicate lookup logic across call sites.

### Decision: Preserve the existing weight lookup contract

Layer code will continue to call `get_weight(name)`, `get_weight_shape(name)`, and related metadata accessors on a single loader object. The loader becomes responsible for routing those requests to the correct mapped shard. This keeps existing model code small and limits the 8B change to the storage boundary.

Alternative considered:
- Replace named access with an iterator over shard-local tensors. Rejected because all existing Qwen3 weight-loading paths are name-based and do not benefit from a broader API change.

### Decision: Treat single-file and sharded checkpoint layouts as equally valid model directories

Model path validation and tests will recognize either `model.safetensors` or `model.safetensors.index.json` as a valid checkpoint entrypoint, provided the expected config/tokenizer files also exist. This allows the same `Qwen3::from_pretrained` surface to work for `0.6B` and `8B`.

Alternative considered:
- Add a separate constructor or CLI path for sharded checkpoints. Rejected because layout is a storage detail, not a model API distinction.

### Decision: Keep shard data memory-mapped and reused within loader lifetime

The sharded loader will not eagerly read all shard data into host memory. It will keep the current `mmap` access model, extended so each shard can be opened and reused when the first tensor from that shard is requested. This minimizes extra copies and keeps the memory profile bounded by the mapped checkpoint files plus runtime tensors.

Alternative considered:
- Eagerly preload all shards into RAM. Rejected because it adds startup cost and duplicates file-backed weight storage without a correctness benefit.

### Decision: Validate 8B support with CPU end-to-end generation, not token-exact assertions

The acceptance target is that `Qwen3-8B` can be loaded and complete generation on CPU using the current runtime. Tests will assert successful end-to-end execution, valid probability outputs, and basic resource sanity, but will not require an exact token sequence match because runtime defaults and sampling paths can vary.

Alternative considered:
- Require deterministic token-by-token equality against transformers. Rejected for this change because it would widen scope into cross-framework parity harnesses and sampling determinism work.

## Risks / Trade-offs

- [Manifest parsing bugs misroute weights to the wrong shard] -> Mitigation: add loader-level tests that fetch tensors from multiple shards and verify shape/data offsets against fixture files.
- [Too many open shard mappings increase file descriptor usage] -> Mitigation: reuse opened mappings per shard path and close them with loader lifetime rather than reopening on every tensor lookup.
- [Single-file `Qwen3-0.6B` regressions] -> Mitigation: preserve the existing direct path and keep model-directory tests for both single-file and sharded layouts.
- [CPU end-to-end 8B generation is slow] -> Mitigation: make correctness the requirement for this change and explicitly defer CPU performance tuning.
- [Future non-Qwen3 models need different shard semantics] -> Mitigation: keep the new loader keyed on standard safetensors index metadata rather than hard-coding Qwen3 layer names.

## Migration Plan

1. Extend `SafeTensors` so it can represent either one file or an index-driven shard set, including manifest parsing and shard reuse.
2. Update Qwen3 model construction and test path helpers to accept either checkpoint layout.
3. Add loader tests for shard parsing and cross-shard tensor lookup alongside existing single-file coverage.
4. Add `Qwen3-8B` CPU model tests or smoke coverage that runs full generation from the local checkpoint path.
5. Keep rollback simple by preserving the current single-file path; if regressions appear, the sharded path can be disabled without removing the single-file implementation.

## Open Questions

- Whether the loader should expose shard-count/debug metadata for test diagnostics, or keep that internal for the first rollout.
- Whether the 8B end-to-end CPU test should live in the existing Qwen3 check suite or a separate, explicitly opt-in smoke test target.
