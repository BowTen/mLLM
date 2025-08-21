#include "op/add.h"

namespace mllm
{
    namespace op
    {
        void Add::forward()
        {
            CHECK(inputs.size() == 2);
            CHECK(outputs.size() == 1);
            kernel::get_mat_add_kernel(device_)(&inputs[0], &inputs[1], &outputs[0], stream_ ? stream_ : nullptr);
        }
        void Add::add(const Tensor &input0, const Tensor &input1, Tensor &output)
        {
            setInput(0, input0);
            setInput(1, input1);
            setOutput(0, output);
            forward();
        }
    }
}