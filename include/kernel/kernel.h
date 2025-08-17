#ifndef MLLM_KERNEL_H
#define MLLM_KERNEL_H

#include "embedding_kernel.h"

namespace mllm
{
    namespace kernel
    {
        typedef void (*EmbeddingKernel)(base::Tensor *input,
                                        base::Tensor *weight,
                                        base::Tensor *output,
                                        size_t vocab_size,
                                        size_t hidden_size,
                                        void *stream);

        EmbeddingKernel get_emb_kernel(base::Device device);
    }
}

#endif // MLLM_KERNEL_H