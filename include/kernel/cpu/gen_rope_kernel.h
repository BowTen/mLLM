#ifndef MLLM_KERNEL_GEN_ROPE_KERNEL_H
#define MLLM_KERNEL_GEN_ROPE_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void gen_rope_kernel_cpu(base::Tensor *inv_freq,
                                 uint32_t l,
                                 uint32_t r,
                                 base::Tensor *cos,
                                 base::Tensor *sin,
                                 void *stream);
    }
}

#endif // MLLM_KERNEL_GEN_ROPE_KERNEL_H