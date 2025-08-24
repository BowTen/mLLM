#ifndef MLLM_KERNEL_ELE_MUL_KERNEL_CUH
#define MLLM_KERNEL_ELE_MUL_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void ele_mul_kernel_cuda(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream);
    }
}

#endif // MLLM_KERNEL_ELE_MUL_KERNEL_CUH