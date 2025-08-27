#include "kernel/cuda/causal_mask_kernel.cuh"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void causal_mask_kernel_cuda_fp32(float *input, size_t head_dim)
        {
            size_t mat_id = blockIdx.x;
            size_t row_id = blockIdx.y;
            size_t seq_len = gridDim.y;
            input += mat_id * seq_len * head_dim + row_id * head_dim;

            for (size_t i = threadIdx.x + head_dim - (seq_len - 1 - row_id); i < head_dim; i += blockDim.x)
            {
                input[i] = -FLOAT_INF;
            }
        }

        void causal_mask_kernel_cuda(base::Tensor *input,
                                     void *stream)
        {
            size_t seq_len = input->shape(-2);
            if (seq_len <= 1)
                return;
            input->contiguous();
            size_t head_dim = input->shape(-1);
            size_t num_mats = input->num_mats();

            dim3 grid(num_mats, seq_len);
            if (stream)
            {
                causal_mask_kernel_cuda_fp32<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(), head_dim);
            }
            else
            {
                LOG(WARNING) << "Causal Mask: use default stream";
                causal_mask_kernel_cuda_fp32<<<grid, 128>>>(input->data(), head_dim);
            }
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}