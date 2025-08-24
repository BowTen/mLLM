#ifndef MLLM_KERNEL_CAUSAL_MASK_KERNEL_CUH
#define MLLM_KERNEL_CAUSAL_MASK_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void causal_mask_kernel_cuda(base::Tensor *input,
                                     void *stream);
    }
}

#endif // MLLM_KERNEL_CAUSAL_MASK_KERNEL_CUH