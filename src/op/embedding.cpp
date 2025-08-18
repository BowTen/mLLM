#include "op/embedding.h"
#include "kernel/kernel.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace op
    {
        Embedding::Embedding(size_t vocab_size, size_t hidden_size, base::Device device)
            : WLayer(1, 1, {vocab_size, hidden_size}, device)
        {
        }

        void Embedding::forward()
        {
            CHECK(!inputs.empty());
            CHECK(!outputs.empty());
            kernel::get_emb_kernel(device_)(&inputs[0], &weight_, &outputs[0], weight_.shape()[0], weight_.shape()[1], nullptr);
        }
    }
}