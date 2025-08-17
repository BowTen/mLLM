#include "op/embedding.h"

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
        }
    }
}