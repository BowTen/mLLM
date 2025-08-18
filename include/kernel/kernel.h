#ifndef MLLM_KERNEL_H
#define MLLM_KERNEL_H

#include <cstddef>
#include "base/tensor.h"

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
        typedef void (*RMSNormKernel)(base::Tensor *input,
                                      base::Tensor *weight,
                                      base::Tensor *output,
                                      float eps,
                                      void *stream);
        typedef void (*AddKernel)(base::Tensor *input0,
                                  base::Tensor *input1,
                                  base::Tensor *output,
                                  void *stream);

        EmbeddingKernel get_emb_kernel(base::Device device);
        RMSNormKernel get_rmsnorm_kernel(base::Device device);
        AddKernel get_add_kernel(base::Device device);
    }
}

#endif // MLLM_KERNEL_H