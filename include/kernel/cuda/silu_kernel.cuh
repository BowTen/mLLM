#ifndef MLLM_KERNEL_SILU_KERNEL_CUH
#define MLLM_KERNEL_SILU_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void silu_kernel_cuda(base::Tensor *input,
                              void *stream);
    }
}

#endif // MLLM_KERNEL_SILU_KERNEL_CUH