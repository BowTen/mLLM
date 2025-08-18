#ifndef MLLM_KERNEL_CUDA_RMS_NORM_KERNEL_CUH
#define MLLM_KERNEL_CUDA_RMS_NORM_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void rms_norm_kernel_cuda(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream);
        void rms_norm_kernel_cuda_vec(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream);
    }
}

#endif // MLLM_KERNEL_CUDA_RMS_NORM_KERNEL_CUH
