#ifndef MLLM_OP_SILU_H
#define MLLM_OP_SILU_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class SiLU : public Layer
        {
        public:
            SiLU(base::Device device, cudaStream_t stream);
            void forward(Tensor &input);
        };
    }
}

#endif // MLLM_OP_SILU_H