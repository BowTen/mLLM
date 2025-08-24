#ifndef MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_CUH
#define MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void last_hidden_state_kernel_cuda(base::Tensor *input,
                                           void *stream);
    }
}

#endif // MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_CUH