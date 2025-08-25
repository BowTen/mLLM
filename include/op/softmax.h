#ifndef MLLM_OP_SOFTMAX_H
#define MLLM_OP_SOFTMAX_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class Softmax : public Layer
        {
        public:
            Softmax(base::Device device, cudaStream_t stream);
            void forward(Tensor &input, Tensor &output);
        };
    }
}

#endif // MLLM_OP_SOFTMAX_H