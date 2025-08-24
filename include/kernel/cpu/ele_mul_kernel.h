#ifndef MLLM_KERNEL_ELE_MUL_KERNEL_H
#define MLLM_KERNEL_ELE_MUL_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void ele_mul_kernel_cpu(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream);
    }
}

#endif // MLLM_KERNEL_ELE_MUL_KERNEL_H