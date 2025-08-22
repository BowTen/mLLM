#ifndef MLLM_KERNEL_ROPE_KERNEL_CUH
#define MLLM_KERNEL_ROPE_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void rope_kernel_cuda(base::Tensor *input,
                              base::Tensor *cos,
                              base::Tensor *sin,
                              base::Tensor *output,
                              void *stream);
    }
}

#endif // MLLM_KERNEL_ROPE_KERNEL_CUH