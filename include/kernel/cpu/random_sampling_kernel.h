#ifndef MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_H
#define MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void random_sampling_kernel_cpu(base::Tensor *probability,
                                        base::Tensor *output,
                                        void *stream);
    }
}

#endif // MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_H