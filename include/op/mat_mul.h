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
            MatMul(base::Device device, cudaStream_t stream)
                : Layer(2, 1, device, stream) {}

            void forward(const Tensor &input0, const Tensor &input1, Tensor &output);
        };
    }
}
#endif // MMLM_OP_MAT_MUL_H