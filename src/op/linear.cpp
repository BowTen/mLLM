#include "op/linear.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace op
    {
        Linear::Linear(std::vector<size_t> shape, base::Device device, cudaStream_t stream)
            : WLayer(1, 1, shape, device, stream)
        {
        }
        void Linear::forward(Tensor &input, Tensor &output)
        {
            setInput(0, input);
            setOutput(0, output);
            kernel::get_mat_mul_kernel(device_)(&inputs[0], &weight_, &outputs[0], stream_);
        }
    }
}
