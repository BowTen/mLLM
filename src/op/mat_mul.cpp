#include "op/mat_mul.h"

namespace mllm
{
    namespace op
    {
        void MatMul::forward(const Tensor &input0, const Tensor &input1, Tensor &output)
        {
            setInput(0, input0);
            setInput(1, input1);
            setOutput(0, output);
            kernel::get_mat_mul_kernel(device_)(&inputs[0], &inputs[1], &outputs[0], stream_);
        }
    }
}