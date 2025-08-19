#ifndef MLLM_MODEL_QWEN3_SELF_ATTN_H
#define MLLM_MODEL_QWEN3_SELF_ATTN_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "op/rms_norm.h"
#include "op/linear.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3SelfAttn : public Layer
        {
            size_t layer_index_;
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

        public:
            Qwen3SelfAttn(size_t layer_index, JsonConfig config, base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward() override;
            using Layer::forward;
            void loadWeight(const std::string &name, base::SafeTensors &st);
        };
    }
}

#endif // MLLM_MODEL_QWEN3_SELF_ATTN_H