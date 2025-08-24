#ifndef MLLM_KERNEL_CAUSAL_MASK_KERNEL_H
#define MLLM_KERNEL_CAUSAL_MASK_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void causal_mask_kernel_cpu(base::Tensor *input,
                                    void *stream);
    }
}

#endif // MLLM_KERNEL_CAUSAL_MASK_KERNEL_H