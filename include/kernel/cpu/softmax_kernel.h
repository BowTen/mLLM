#ifndef MLLM_KERNEL_SOFTMAX_KERNEL_H
#define MLLM_KERNEL_SOFTMAX_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void softmax_kernel_cpu(base::Tensor *input,
                                base::Tensor *output,
                                void *stream);
    }
}

#endif // MLLM_KERNEL_SOFTMAX_KERNEL_H