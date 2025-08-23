#ifndef MMLM_OP_MAT_MUL_H
#define MMLM_OP_MAT_MUL_H

#include "kernel/kernel.h"
#include "layer.h"
namespace mllm
{
    namespace op
    {
        class MatMul : public Layer
        {
        public:
            MatMul(base::Device device = base::Device::CPU, cudaStream_t stream = nullptr)
                : Layer(2, 1, device, stream) {}

            void forward() override;
            using Layer::forward;
            void matmul(const Tensor &input0, const Tensor &input1, Tensor &output);
        };
    }
}
#endif // MMLM_OP_MAT_MUL_H