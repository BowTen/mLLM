#ifndef MLLM_OP_LINEAR_H
#define MLLM_OP_LINEAR_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class Linear : public WLayer
        {

        public:
            Linear(std::vector<size_t> shape, base::Device device, cudaStream_t stream);
            void forward(Tensor &input, Tensor &output);
        };
    }
}

#endif // MLLM_OP_LINEAR_H