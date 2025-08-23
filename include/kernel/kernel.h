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
                                        size_t hidden_size,
                                        void *stream);
        typedef void (*RMSNormKernel)(base::Tensor *input,
                                      base::Tensor *weight,
                                      base::Tensor *output,
                                      float eps,
                                      void *stream);
        typedef void (*MatAddKernel)(base::Tensor *input0,
                                     base::Tensor *input1,
                                     base::Tensor *output,
                                     void *stream);
        typedef void (*MatMulKernel)(base::Tensor *input0,
                                     base::Tensor *input1,
                                     base::Tensor *output,
                                     void *stream);
        typedef void (*ContiguousKernel)(base::Tensor *input,
                                         void *stream);
        typedef void (*GenRoPEKernel)(base::Tensor *inv_freq,
                                      uint32_t pos_start,
                                      uint32_t pos_end,
                                      base::Tensor *cos,
                                      base::Tensor *sin,
                                      void *stream);
        typedef void (*RoPEKernel)(base::Tensor *input,
                                   base::Tensor *cos,
                                   base::Tensor *sin,
                                   base::Tensor *output,
                                   void *stream);
        typedef void (*SoftmaxKernel)(base::Tensor *input,
                                      base::Tensor *output,
                                      void *stream);

        EmbeddingKernel get_emb_kernel(base::Device device);
        RMSNormKernel get_rmsnorm_kernel(base::Device device);
        MatAddKernel get_mat_add_kernel(base::Device device);
        MatMulKernel get_mat_mul_kernel(base::Device device);
        ContiguousKernel get_contiguous_kernel(base::Device device);
        RoPEKernel get_rope_kernel(base::Device device);
        GenRoPEKernel get_gen_rope_kernel(base::Device device);
        SoftmaxKernel get_softmax_kernel(base::Device device);
    }
}

#endif // MLLM_KERNEL_H