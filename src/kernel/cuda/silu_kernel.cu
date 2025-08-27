#include "kernel/cuda/silu_kernel.cuh"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void silu_kernel_cuda_fp32(float *input, size_t seq_len, size_t head_dim)
        {
            size_t mat_id = blockIdx.x;
            size_t row_id = blockIdx.y;

            input += mat_id * seq_len * head_dim + row_id * head_dim;

            float4 *input_vec = reinterpret_cast<float4 *>(input);
            size_t vec_size = head_dim / 4;
            size_t vec_end = vec_size * 4;

            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 v = input_vec[i];
                v.x = v.x / (1.0f + expf(-v.x));
                v.y = v.y / (1.0f + expf(-v.y));
                v.z = v.z / (1.0f + expf(-v.z));
                v.w = v.w / (1.0f + expf(-v.w));
                input_vec[i] = v;
            }
            for (size_t i = threadIdx.x + vec_end; i < head_dim; i += blockDim.x)
            {
                input[i] = input[i] / (1.0f + expf(-input[i]));
            }
        }

        void silu_kernel_cuda(base::Tensor *input, [[maybe_unused]] void *stream)
        {
            size_t seq_len = input->shape(-2);
            input->contiguous();
            size_t head_dim = input->shape(-1);
            size_t num_mats = input->num_mats();

            dim3 grid(num_mats, seq_len);
            if (stream)
            {
                silu_kernel_cuda_fp32<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(), seq_len, head_dim);
            }
            else
            {
                silu_kernel_cuda_fp32<<<grid, 128>>>(input->data(), seq_len, head_dim);
            }
            CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}