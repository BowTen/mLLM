#ifndef MLLM_OP_EMBEDDING_H
#define MLLM_OP_EMBEDDING_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class Embedding : public WLayer
        {
        public:
            Embedding(size_t vocab_size, size_t hidden_size, base::Device device = base::Device::CPU);
            void forward() override;
        };
    }
}

#endif // MLLM_OP_EMBEDDING_H