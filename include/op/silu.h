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
            SiLU(base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward() override;
            using Layer::forward;
        };
    }
}

#endif // MLLM_OP_SILU_H