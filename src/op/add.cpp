#include "op/add.h"

namespace mllm
{
    namespace op
    {
        void Add::forward()
        {
            CHECK(inputs.size() == 2);
            CHECK(outputs.size() == 1);
            if (device_ == base::Device::CUDA)
            {
                CHECK(stream_ != nullptr) << "CUDA stream must be set for CUDA device.";
            }
            kernel::get_add_kernel(device_)(&inputs[0], &inputs[1], &outputs[0], stream_ ? stream_ : nullptr);
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