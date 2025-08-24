#include "kernel/kernel.h"
#include "kernel/cpu/embedding_kernel.h"
#include "kernel/cpu/rms_norm_kernel.h"
#include "kernel/cpu/mat_add_kernel.h"
#include "kernel/cpu/mat_mul_kernel.h"
#include "kernel/cpu/ele_mul_kernel.h"
#include "kernel/cpu/contiguous_kernel.h"
#include "kernel/cpu/rope_kernel.h"
#include "kernel/cpu/gen_rope_kernel.h"
#include "kernel/cpu/softmax_kernel.h"
#include "kernel/cpu/causal_mask_kernel.h"
#include "kernel/cpu/silu_kernel.h"
#include "kernel/cpu/last_hidden_state_kernel.h"
#include "kernel/cuda/embedding_kernel.cuh"
#include "kernel/cuda/rms_norm_kernel.cuh"
#include "kernel/cuda/mat_add_kernel.cuh"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/ele_mul_kernel.cuh"
#include "kernel/cuda/contiguous_kernel.cuh"
#include "kernel/cuda/rope_kernel.cuh"
#include "kernel/cuda/gen_rope_kernel.cuh"
#include "kernel/cuda/softmax_kernel.cuh"
#include "kernel/cuda/causal_mask_kernel.cuh"
#include "kernel/cuda/silu_kernel.cuh"
#include "kernel/cuda/last_hidden_state_kernel.cuh"
#include <stdexcept>
#include "base/util.h"

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

        EleMulKernel get_ele_mul_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return ele_mul_kernel_cpu;
            case base::Device::CUDA:
                return ele_mul_kernel_cuda;
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

        CausalMaskKernel get_causal_mask_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return causal_mask_kernel_cpu;
            case base::Device::CUDA:
                return causal_mask_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }
        SiLUKernel get_silu_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return silu_kernel_cpu;
            case base::Device::CUDA:
                return silu_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        LastHiddenStateKernel get_last_hidden_state_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return last_hidden_state_kernel_cpu;
            case base::Device::CUDA:
                return last_hidden_state_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        void random_sampling_cpu(base::Tensor *probability, base::Tensor *token, base::Device device)
        {
            LOG(WARNING) << "random_sampling_cpu, it's just for development, TODO: Implement cuda operators";
            probability->toDevice(base::Device::CPU);
            token->toDevice(base::Device::CPU);
            float rand_num = base::get_random_float();

            float eps = 1e-6;
            size_t vocab_size = probability->shape(-1);
            float *prob_data = probability->data();
            VLOG(DEBUG) << "Random sampling with rand float: " << rand_num;
            for (uint32_t i = 0; i < vocab_size; i++)
            {
                rand_num -= prob_data[i];
                if (rand_num < eps)
                {
                    reinterpret_cast<uint32_t *>(token->data())[0] = i;
                    break;
                }
            }
            CHECK_LT(rand_num, eps);

            probability->toDevice(device);
            token->toDevice(device);
        }
    }
}