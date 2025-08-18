#include "kernel/kernel.h"
#include "kernel/cpu/embedding_kernel.h"
#include "kernel/cuda/embedding_kernel.cuh"
#include <stdexcept>

namespace mllm
{
    namespace kernel
    {
        EmbeddingKernel get_emb_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return emb_kernel_cpu;
            case base::Device::CUDA:
                return emb_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }
    }
}