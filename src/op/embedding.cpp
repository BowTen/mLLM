#include "op/embedding.h"
#include "kernel/kernel.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace op
    {
        Embedding::Embedding(size_t vocab_size, size_t hidden_size, base::Device device, cudaStream_t stream)
            : WLayer(1, 1, {vocab_size, hidden_size}, device, stream)
        {
        }

        void Embedding::forward(Tensor &input, Tensor &output)
        {
            setInput(0, input);
            setOutput(0, output);
            if (device_ == base::Device::CUDA)
            {
                CHECK(stream_ != nullptr) << "CUDA stream must be set for CUDA device.";
            }
            kernel::get_emb_kernel(device_)(&inputs[0], &weight_, &outputs[0], weight_.shape(-1), stream_);
        }
    }
}