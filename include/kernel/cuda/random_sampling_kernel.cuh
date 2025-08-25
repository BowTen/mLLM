#ifndef MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_CUH
#define MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void random_sampling_kernel_cuda(base::Tensor *probability,
                                         base::Tensor *output,
                                         void *stream);
    }
}

#endif // MLLM_KERNEL_RANDOM_SAMPLING_KERNEL_CUH