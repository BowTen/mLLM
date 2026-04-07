#ifndef MLLM_KERNEL_SOFTMAX_KERNEL_CUH
#define MLLM_KERNEL_SOFTMAX_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        bool cuda_softmax_library_available();
        void softmax_kernel_cuda_handwritten(base::Tensor *input,
                                             base::Tensor *output,
                                             void *stream);
        void softmax_kernel_cuda_library_first(base::Tensor *input,
                                               base::Tensor *output,
                                               void *stream);
    }
}

#endif // MLLM_KERNEL_SOFTMAX_KERNEL_CUH
