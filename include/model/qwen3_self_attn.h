#ifndef MLLM_MODEL_QWEN3_SELF_ATTN_H
#define MLLM_MODEL_QWEN3_SELF_ATTN_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "op/rms_norm.h"
#include "op/linear.h"
#include "op/mat_mul.h"
#include "op/causal_mask.h"
#include "op/softmax.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3SelfAttn
        {
            size_t layer_index_;
            base::Device device_;
            cudaStream_t stream_;
            JsonConfig config_;
            size_t hidden_size;
            size_t head_dim;
            size_t num_attention_heads;
            size_t num_key_value_heads;
            Linear q_proj;
            Linear k_proj;
            Linear v_proj;
            Linear o_proj;
            RMSNorm q_norm;
            RMSNorm k_norm;
            MatMul mat_mul;
            MatMul mat_sc_mul;
            MatMul mat_mul_attn_output;
            CausalMask causal_mask;
            Softmax softmax;
            std::string name_;

            Tensor q_output;
            Tensor k_output;
            Tensor v_output;
            Tensor k_cache;
            Tensor v_cache;
            Tensor scaling;

            void repeat_kv(Tensor &k, Tensor &v);

        public:
            Qwen3SelfAttn(size_t layer_index, JsonConfig config, base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward(Tensor *hidden_state, Tensor *output, base::PosEmb position_embeddings);
            void loadWeight(const std::string &name, base::SafeTensors &st);

            std::vector<WLayer *> weighted_layers();
        };
    }
}

#endif // MLLM_MODEL_QWEN3_SELF_ATTN_H