#include "kernel/kernel.h"
#include "kernel/cpu/embedding_kernel.h"
#include "kernel/cpu/rms_norm_kernel.h"
#include "kernel/cpu/add_kernel.h"
#include "kernel/cpu/mat_mul_kernel.h"
#include "kernel/cpu/contiguous_kernel.h"
#include "kernel/cuda/embedding_kernel.cuh"
#include "kernel/cuda/rms_norm_kernel.cuh"
#include "kernel/cuda/add_kernel.cuh"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/contiguous_kernel.cuh"
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
                return emb_kernel_cuda_vec; // 向量化存取版本
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        RMSNormKernel get_rmsnorm_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return rms_norm_kernel_cpu;
            case base::Device::CUDA:
                return rms_norm_kernel_cuda_vec; // 向量化存取版本
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        AddKernel get_add_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return add_kernel_cpu;
            case base::Device::CUDA:
                return add_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        MatMulKernel get_matmul_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return mat_mul_kernel_cpu;
            case base::Device::CUDA:
                return mat_mul_kernel_cuda_vec; // 向量化存取版本
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        ContiguousKernel get_contiguous_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return contiguous_kernel_cpu;
            case base::Device::CUDA:
                return contiguous_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }
    }
}