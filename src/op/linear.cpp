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
        void Linear::forward()
        {
            CHECK(!inputs.empty());
            CHECK(!outputs.empty());
            kernel::get_matmul_kernel(device_)(&inputs[0], &weight_, &outputs[0], stream_);
        }
    }
}
