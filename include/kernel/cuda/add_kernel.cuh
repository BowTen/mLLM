#ifndef MLLM_KERNEL_CUDA_ADD_KERNEL_CUH
#define MLLM_KERNEL_CUDA_ADD_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void add_kernel_cuda(base::Tensor *input0,
                             base::Tensor *input1,
                             base::Tensor *output,
                             void *stream);
    }
}

#endif // MLLM_KERNEL_CUDA_ADD_KERNEL_CUH