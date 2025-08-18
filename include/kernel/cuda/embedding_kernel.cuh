#ifndef MLLM_EMBEDDING_KERNEL_CUH
#define MLLM_EMBEDDING_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void emb_kernel_cuda(base::Tensor *input,
                             base::Tensor *weight,
                             base::Tensor *output,
                             size_t vocab_size,
                             size_t hidden_size,
                             void *stream);
        void emb_kernel_cuda_vec(base::Tensor *input,
                                 base::Tensor *weight,
                                 base::Tensor *output,
                                 size_t vocab_size,
                                 size_t hidden_size,
                                 void *stream);
    }
}

#endif // MLLM_EMBEDDING_KERNEL_CUH