#ifndef MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_H
#define MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_H

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void last_hidden_state_kernel_cpu(base::Tensor *input,
                                          void *stream);
    }
}

#endif // MLLM_KERNEL_LAST_HIDDEN_STATE_KERNEL_H