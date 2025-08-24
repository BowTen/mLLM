#ifndef MLLM_OP_RMS_NORM_H
#define MLLM_OP_RMS_NORM_H

#include "layer.h"

namespace mllm
{
    namespace op
    {
        class RMSNorm : public WLayer
        {
            float eps_;

        public:
            RMSNorm(size_t hidden_size, float eps, base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward(Tensor &input, Tensor &output);
        };
    }
}

#endif // MLLM_OP_RMS_NORM_H