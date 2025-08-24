#include "op/causal_mask.h"
#include "op/layer.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace op
    {
        CausalMask::CausalMask(base::Device device, cudaStream_t stream)
            : Layer(1, 0, device, stream)
        {
        }

        void CausalMask::forward(Tensor &input)
        {
            setInput(0, input);
            kernel::get_causal_mask_kernel(device_)(&inputs[0], stream_);
        }
    }
}