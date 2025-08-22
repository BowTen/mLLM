#ifndef MLLM_KERNEL_GEN_ROPE_KERNEL_CUH
#define MLLM_KERNEL_GEN_ROPE_KERNEL_CUH

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        void gen_rope_kernel_cuda(base::Tensor *inv_freq,
                                  uint32_t l,
                                  uint32_t r,
                                  base::Tensor *cos,
                                  base::Tensor *sin,
                                  void *stream);
    }
}

#endif // MLLM_KERNEL_GEN_ROPE_KERNEL_CUH