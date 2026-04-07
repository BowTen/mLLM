# CUDA Library Operator Rollout

This note records the task 4.1 evaluation for follow-on CUDA operators after the matrix multiplication rollout.

## Current Baseline

- Matrix multiplication is already library-first on eligible 2D CUDA tensors.
- `softmax` and `rms_norm` still dispatch directly to handwritten CUDA kernels through [kernel.cpp](/home/zz/workspace/mLLM/src/kernel/kernel.cpp).
- Any operator promoted beyond handwritten-only execution must preserve the project's explicit stream behavior and keep a predictable handwritten fallback.

## Softmax Evaluation

### Current CUDA Contract

- Entry point: [softmax_kernel_cuda](/home/zz/workspace/mLLM/src/kernel/cuda/softmax_kernel.cu)
- Dispatch surface: `kernel::get_softmax_kernel(Device::CUDA)`
- Logical behavior: row-wise softmax over the last dimension
- Shape handling:
  - treats all leading dimensions as `num_mats`
  - uses `shape(-2)` as row count within each logical matrix
  - uses `shape(-1)` as the normalized axis
- Layout assumptions:
  - input and output shapes must match
  - the CUDA kernel assumes contiguous dense storage even though it does not currently call `contiguous()` or validate contiguity explicitly
  - vectorized path requires float4-compatible last-dimension alignment
- Stream semantics:
  - supports explicit CUDA streams
  - falls back to the default stream when `stream == nullptr`
  - current unit tests mostly cover the default-stream path, while real Qwen3 CUDA execution uses a non-null model stream
- In-place behavior:
  - existing call sites use `softmax.forward(t, t)`, so any library-backed rollout must preserve input/output aliasing semantics
- Numerical behavior:
  - current implementation computes `exp(x) / sum(exp(x))` directly
  - there is no max-subtraction stabilization pass today

### Qwen3 Usage

- Attention probabilities in [qwen3_self_attn.cpp](/home/zz/workspace/mLLM/src/model/qwen3_self_attn.cpp)
  - tensor shape is effectively `{num_attention_heads, q_seq, k_seq}`
  - softmax is applied after causal masking
- Final token probabilities in [qwen3.cpp](/home/zz/workspace/mLLM/src/model/qwen3.cpp)
  - tensor shape is effectively `{1, vocab_size}`

### Rollout Recommendation

Softmax is approved as the next candidate operator for task 4.2, with this rollout contract:

- CUDA only
- contiguous dense tensors only
- normalize over the last dimension only
- preserve explicit stream forwarding
- preserve in-place `input == output` behavior
- keep causal masking as a separate handwritten/operator step; do not require fused masked-softmax
- only route layouts that are already contiguous and safe for both the library path and the existing handwritten fallback
- do not promise a non-contiguous handwritten fallback unless task 4.2 adds an explicit `contiguous()` materialization step or a true stride-aware fallback

Why softmax is the best next step:

- the operator boundary is already narrow and well-defined
- Qwen3 uses softmax in two important paths with stable row-wise semantics
- the rollout contract can be stated without redesigning tensor layout or model code
- a library-backed softmax can be evaluated independently from masking, attention layout, and KV-cache structure

### Validation Rule For Task 4.2

- The current handwritten CPU/CUDA baseline uses direct `exp(x) / sum(exp(x))` without max-subtraction stabilization.
- Task 4.2 does not need to preserve that implementation detail bit-for-bit.
- A library-backed softmax may use a numerically stabilized implementation, but validation must use tolerance-based parity against the current handwritten path on representative Qwen3 attention and logits shapes.
- Validation should include large-logit cases explicitly so behavior changes are intentional rather than accidental.

## RMSNorm Evaluation

### Current CUDA Contract

- Entry point: [rms_norm_kernel_cuda](/home/zz/workspace/mLLM/src/kernel/cuda/rms_norm_kernel.cu)
- Dispatch surface: `kernel::get_rmsnorm_kernel(Device::CUDA)`
- Logical behavior: normalize each row over the last dimension, then apply a learned weight vector
- Shape handling:
  - input and output shapes must match
  - weight uses `shape(-1)` as the hidden dimension
  - leading dimensions are flattened as `num_mats * seq_size`
- Layout assumptions:
  - input, weight, and output are made contiguous before launch
  - vectorized execution assumes float4-compatible hidden-size alignment
- Validation gap:
  - the current CUDA kernel trusts `weight->shape(-1)` as `hidden_size`
  - it does not explicitly validate `weight->shape(-1) == input->shape(-1)`
- Stream semantics:
  - supports explicit CUDA streams
  - falls back to the default stream when `stream == nullptr`

### Qwen3 Usage

- Final model norm in [qwen3.cpp](/home/zz/workspace/mLLM/src/model/qwen3.cpp)
- Decode-layer input and post-attention norms in [qwen3_decode_layer.cpp](/home/zz/workspace/mLLM/src/model/qwen3_decode_layer.cpp)
- Attention `q_norm` and `k_norm` in [qwen3_self_attn.cpp](/home/zz/workspace/mLLM/src/model/qwen3_self_attn.cpp)

### Rollout Recommendation

RMSNorm remains handwritten-only for this change.

Reasons to defer:

- it is more pervasive in Qwen3 than softmax, so mistakes affect more of the model
- the current repository has no library integration scaffolding specific to normalization kernels beyond generic CUDA dependency detection
- this change has no benchmark evidence yet showing a clear library win for RMSNorm on the project's actual shapes
- promoting RMSNorm first would require a tighter numerical contract because it appears in residual-critical paths and in attention head normalization
- the repository already has CPU/CUDA checks that touch Qwen3 RMSNorm sites, including `q_norm` / `k_norm`, but that coverage is still indirect and not yet strong enough to serve as rollout-grade evidence on its own

If RMSNorm is revisited later, the minimum contract should be:

- CUDA only
- contiguous dense tensors only
- normalization over the last dimension only
- require `weight->shape(-1) == input->shape(-1)` validation before dispatch
- explicit stream forwarding
- preserve in-place `input == output` behavior
- handwritten fallback retained
- parity validation across both hidden-size `1024` and head-dim `128` Qwen3 cases

## Task 4.2 Decision

The next approved operator for implementation is **softmax**.

Task 4.2 should therefore:

- add a library-first softmax selector/dispatcher at the kernel layer
- preserve the current handwritten CUDA softmax as fallback
- keep masking separate from softmax
- validate both attention softmax and final-logit softmax paths against the existing handwritten implementation
