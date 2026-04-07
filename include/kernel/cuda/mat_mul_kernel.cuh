#ifndef MLLM_KERNEL_MAT_MUL_KERNEL_CUH
#define MLLM_KERNEL_MAT_MUL_KERNEL_CUH

#include "base/tensor.h"
#include "kernel/cuda/mat_mul_backend_selector.h"

namespace mllm
{
    namespace kernel
    {
        void mat_mul_kernel_cuda_library_first(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream);
        void mat_mul_kernel_cuda_vec(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream);
        void mat_mul_kernel_cuda_cublas(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream);
    }
}

#endif // MLLM_KERNEL_MAT_MUL_KERNEL_CUH
