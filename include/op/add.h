#ifndef MMLM_OP_ADD_H
#define MMLM_OP_ADD_H

#include "kernel/kernel.h"
#include "layer.h"
namespace mllm
{
    namespace op
    {
        class Add : public Layer
        {
        public:
            Add(base::Device device = base::Device::CPU, cudaStream_t stream = nullptr)
                : Layer(2, 1, device, stream) {}

            void forward(const Tensor &input0, const Tensor &input1, Tensor &output);
        };
    }
}
#endif // MMLM_OP_ADD_H