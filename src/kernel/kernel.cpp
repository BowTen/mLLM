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
#include "kernel/cpu/random_sampling_kernel.h"
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
#include "kernel/cuda/random_sampling_kernel.cuh"
#include "kernel/cuda/gemm_kernel.cuh"
#include <stdexcept>
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        bool align_float4(base::Tensor *tensor)
        {
            return tensor->shape(-1) % 4 == 0 || tensor->shape(-1) == tensor->size();
        }

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
                return mat_mul_kernel_cuda_vec;
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

        RandomSamplingKernel get_random_sampling_kernel(base::Device device)
        {
            switch (device)
            {
            case base::Device::CPU:
                return random_sampling_kernel_cpu;
            case base::Device::CUDA:
                return random_sampling_kernel_cuda;
            default:
                throw std::runtime_error("Unsupported device");
            }
        }

        void sampling_kernel(base::Tensor *probability,
                             base::Tensor *output,
                             size_t top_k,
                             float top_p,
                             float min_p)
        {
            auto orig_device = probability->device();
            probability->toDevice(base::Device::CPU);
            output->toDevice(base::Device::CPU);

            std::vector<std::pair<float, uint32_t>> prob_data;
            float *data = probability->data();
            size_t vocab_size = probability->logic_size();
            for (size_t i = 0; i < vocab_size; i++)
            {
                prob_data.emplace_back(data[i], i);
            }

            std::sort(prob_data.begin(), prob_data.end(), std::greater<std::pair<float, uint32_t>>());
            if (top_k > 0 && top_k < prob_data.size())
            {
                prob_data.erase(prob_data.begin() + top_k, prob_data.end());
            }
            for (size_t i = 0; i < prob_data.size(); i++)
            {
                top_p -= prob_data[i].first;
                if (top_p <= 0)
                {
                    prob_data.erase(prob_data.begin() + i + 1, prob_data.end());
                    break;
                }
            }
            min_p *= prob_data[0].first;
            for (size_t i = 0; i < prob_data.size(); i++)
            {
                if (prob_data[i].first < min_p)
                {
                    prob_data.erase(prob_data.begin() + i, prob_data.end());
                    break;
                }
            }

            float rand_num = base::get_random_float();
            float eps = 1e-6;
            for (size_t i = 0; i < prob_data.size(); i++)
            {
                rand_num -= prob_data[i].first;
                if (rand_num <= eps)
                {
                    reinterpret_cast<uint32_t *>(output->data())[0] = prob_data[i].second;
                    break;
                }
            }

            probability->toDevice(orig_device);
            output->toDevice(orig_device);
        }
    }
}