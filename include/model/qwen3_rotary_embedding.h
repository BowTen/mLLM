#ifndef MLLM_MODEL_QWEN3_ROTARY_EMBEDDING_H
#define MLLM_MODEL_QWEN3_ROTARY_EMBEDDING_H

#include "op/layer.h"

namespace mllm
{
    namespace model
    {

        class Qwen3RotaryEmbedding
        {
        private:
            base::Device device_;
            cudaStream_t stream_;
            size_t head_dim;
            float rope_theta;
            base::Tensor inv_freq;

        public:
            Qwen3RotaryEmbedding(op::JsonConfig config,
                                 base::Device device,
                                 cudaStream_t stream);
            void forward(size_t pos_start, size_t pos_end, base::PosEmb pos_emb);
        };
    }
}

#endif // MLLM_MODEL_QWEN3_ROTARY_EMBEDDING_H