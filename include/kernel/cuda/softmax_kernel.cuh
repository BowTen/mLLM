#ifndef MLLM_KERNEL_SOFTMAX_KERNEL_CUH
#define MLLM_KERNEL_SOFTMAX_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void softmax_kernel_cuda(base::Tensor *input,
                                 base::Tensor *output,
                                 void *stream);
    }
}

#endif // MLLM_KERNEL_SOFTMAX_KERNEL_CUH