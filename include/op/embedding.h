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
            Embedding(size_t vocab_size, size_t hidden_size, base::Device device, cudaStream_t stream, base::DType dtype = base::default_float_dtype());
            void forward(Tensor &input, Tensor &output);
        };
    }
}

#endif // MLLM_OP_EMBEDDING_H
