#include "op/rms_norm.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace op
    {
        RMSNorm::RMSNorm(size_t hidden_size, float eps, base::Device device, cudaStream_t stream)
            : WLayer(1, 1, {hidden_size}, device, stream), eps_(eps)
        {
        }

        void RMSNorm::forward()
        {
            CHECK(!inputs.empty());
            CHECK(!outputs.empty());
            if (device_ == base::Device::CUDA)
            {
                CHECK(stream_ != nullptr) << "CUDA stream must be set for CUDA device.";
            }
            kernel::get_rmsnorm_kernel(device_)(&inputs[0], &weight_, &outputs[0], eps_, stream_ ? stream_ : nullptr);
        }
    }
}
