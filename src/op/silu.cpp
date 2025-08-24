#include "op/silu.h"
#include "op/layer.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace op
    {
        SiLU::SiLU(base::Device device, cudaStream_t stream)
            : Layer(1, 0, device, stream)
        {
        }

        void SiLU::forward(Tensor &input)
        {
            setInput(0, input);
            kernel::get_silu_kernel(device_)(&inputs[0], stream_);
        }
    }
}