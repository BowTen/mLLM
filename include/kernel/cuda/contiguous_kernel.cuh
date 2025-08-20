#ifndef MLLM_KERNEL_CONTIGUOUS_KERNEL_CUH
#define MLLM_KERNEL_CONTIGUOUS_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void contiguous_kernel_cuda(base::Tensor *input,
                                    void *stream);
    }
}

#endif // MLLM_KERNEL_CONTIGUOUS_KERNEL_CUH