#include "op/softmax.h"
#include "op/layer.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace op
    {
        Softmax::Softmax(base::Device device, cudaStream_t stream)
            : Layer(1, 1, device, stream)
        {
        }

        void Softmax::forward(Tensor &input, Tensor &output)
        {
            setInput(0, input);
            setOutput(0, output);
            kernel::get_softmax_kernel(device_)(&inputs[0], &outputs[0], stream_);
        }
    }
}