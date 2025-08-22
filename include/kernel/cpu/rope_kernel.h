#ifndef MLLM_KERNEL_ROPE_KERNEL_H
#define MLLM_KERNEL_ROPE_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void rope_kernel_cpu(base::Tensor *input,
                             base::Tensor *cos,
                             base::Tensor *sin,
                             base::Tensor *output,
                             void *stream);
    }
}

#endif // MLLM_KERNEL_ROPE_KERNEL_H