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
            Linear(std::vector<size_t> shape, base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward() override;
        };
    }
}

#endif // MLLM_OP_LINEAR_H