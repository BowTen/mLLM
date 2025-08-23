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
            Softmax(base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward() override;
            using Layer::forward;
        };
    }
}

#endif // MLLM_OP_SOFTMAX_H