#include "kernel/cuda/random_sampling_kernel.cuh"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void write_u32(uint32_t *data, uint32_t value)
        {
            *data = value;
        }

        // TODO: 并行前缀和优化
        void random_sampling_kernel_cuda(base::Tensor *probability,
                                         base::Tensor *output,
                                         void *stream)
        {
            probability->toDevice(base::Device::CPU);
            float rand_num = base::get_random_float();

            float eps = 1e-6;
            size_t vocab_size = probability->shape(-1);
            float *prob_data = probability->data();
            size_t token_id = 0;
            for (uint32_t i = 0; i < vocab_size; i++)
            {
                rand_num -= prob_data[i];
                if (rand_num < eps)
                {
                    token_id = i;
                    break;
                }
            }
            CHECK_LT(rand_num, eps);
            if (stream)
            {
                write_u32<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(reinterpret_cast<uint32_t *>(output->data()), token_id);
            }
            else
            {
                LOG(WARNING) << "Random sampling kernel called without stream";
                write_u32<<<1, 1>>>(reinterpret_cast<uint32_t *>(output->data()), token_id);
            }
            probability->toDevice(base::Device::CUDA);

            // CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}