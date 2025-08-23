#include "kernel/kernel.h"
#include "kernel/cpu/embedding_kernel.h"
#include "kernel/cpu/rms_norm_kernel.h"
#include "kernel/cpu/mat_add_kernel.h"
#include "kernel/cpu/mat_mul_kernel.h"
#include "kernel/cpu/contiguous_kernel.h"
#include "kernel/cpu/rope_kernel.h"
#include "kernel/cpu/gen_rope_kernel.h"
#include "kernel/cpu/softmax_kernel.h"
#include "kernel/cuda/embedding_kernel.cuh"
#include "kernel/cuda/rms_norm_kernel.cuh"
#include "kernel/cuda/mat_add_kernel.cuh"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/contiguous_kernel.cuh"
#include "kernel/cuda/rope_kernel.cuh"
#include "kernel/cuda/gen_rope_kernel.cuh"
#include "kernel/cuda/softmax_kernel.cuh"
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
                return emb_kernel_cuda; // 向量化存取版本
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
                return rms_norm_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        MatAddKernel get_mat_add_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return mat_add_kernel_cpu;
            case base::Device::CUDA:
                return mat_add_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        MatMulKernel get_mat_mul_kernel(base::Device device)
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

        RoPEKernel get_rope_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return rope_kernel_cpu;
            case base::Device::CUDA:
                return rope_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        GenRoPEKernel get_gen_rope_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return gen_rope_kernel_cpu;
            case base::Device::CUDA:
                return gen_rope_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        SoftmaxKernel get_softmax_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return softmax_kernel_cpu;
            case base::Device::CUDA:
                return softmax_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

    }
}