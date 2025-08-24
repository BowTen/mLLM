#ifndef MLLM_OP_CAUSAL_MASK_H
#define MLLM_OP_CAUSAL_MASK_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class CausalMask : public Layer
        {
        public:
            CausalMask(base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward(Tensor &input);
        };
    }
}

#endif // MLLM_OP_CAUSAL_MASK_H