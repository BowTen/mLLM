#include "op/mat_mul.h"

namespace mllm
{
    namespace op
    {
        void MatMul::forward()
        {
            CHECK(inputs.size() == 2);
            CHECK(outputs.size() == 1);
            kernel::get_mat_mul_kernel(device_)(&inputs[0], &inputs[1], &outputs[0], stream_);
        }
        void MatMul::matmul(const Tensor &input0, const Tensor &input1, Tensor &output)
        {
            setInput(0, input0);
            setInput(1, input1);
            setOutput(0, output);
            forward();
        }
    }
}