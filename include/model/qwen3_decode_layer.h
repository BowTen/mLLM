#ifndef MLLM_MODEL_QWEN3_DECODE_LAYER_H
#define MLLM_MODEL_QWEN3_DECODE_LAYER_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "base/common.h"
#include "op/embedding.h"
#include "op/rms_norm.h"
#include "op/add.h"
#include "qwen3_self_attn.h"
#include "qwen3_mlp.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3DecodeLayer
        {
        private:
            size_t layer_index_;
            RMSNorm input_layernorm;
            RMSNorm post_attention_layernorm;
            Qwen3SelfAttn self_attn;
            Qwen3MLP mlp;
            Add add_op;
            std::string name_;

        public:
            Qwen3DecodeLayer(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream);
            void forward(Tensor *input, Tensor *output, base::PosEmb position_embeddings);
            void loadWeight(const std::string &name, base::SafeTensors &st);

            std::vector<WLayer *> weighted_layers();
        };
    }
}

#endif // MLLM_MODEL_QWEN3_DECODE_LAYER_H