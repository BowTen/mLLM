#ifndef MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H
#define MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void rms_norm_kernel_cpu(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream);
    }
}

#endif // MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H