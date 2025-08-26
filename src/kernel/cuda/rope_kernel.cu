#include "kernel/cuda/rope_kernel.cuh"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void rope_kernel_cuda_fp32(float *input,
                                              float *cos,
                                              float *sin,
                                              float *output,
                                              size_t head_dim)
        {
            size_t head_id = blockIdx.x;
            size_t token_id = blockIdx.y;
            size_t seq_len = gridDim.y;
            size_t stride = head_dim / 2;

            input += head_id * seq_len * head_dim + token_id * head_dim;
            output += head_id * seq_len * head_dim + token_id * head_dim;
            cos += token_id * head_dim;
            sin += token_id * head_dim;

            for (size_t i = threadIdx.x; i < stride; i += blockDim.x)
            {
                size_t j = i + stride;
                float cos_val = cos[i];
                float sin_val = sin[i];
                float input_i = input[i];
                float input_j = input[j];
                output[i] = input_i * cos_val - input_j * sin_val;
                output[j] = input_i * sin_val + input_j * cos_val;
            }
        }

        void rope_kernel_cuda(base::Tensor *input,
                              base::Tensor *cos,
                              base::Tensor *sin,
                              base::Tensor *output,
                              void *stream)
        {
            CHECK(input->shape(-1) % 2 == 0) << "head dim must be even";
            CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
            CHECK(cos->shape() == sin->shape()) << "Cosine and sine shapes must be the same.";
            CHECK(input->shape(-1) == cos->shape(-1) && input->shape(-2) == cos->shape(-2));
            input->contiguous();
            cos->contiguous();
            sin->contiguous();
            output->contiguous();
            size_t head_dim = input->shape(-1);
            size_t num_heads = input->num_mats();
            size_t seq_len = input->shape(-2);

            dim3 grid(num_heads, seq_len);
            if (stream)
            {
                rope_kernel_cuda_fp32<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(),
                                                                                           cos->data(),
                                                                                           sin->data(),
                                                                                           output->data(),
                                                                                           head_dim);
            }
            else
            {
                LOG(WARNING) << "RoPR: Use default stream";
                rope_kernel_cuda_fp32<<<grid, 128>>>(input->data(),
                                                     cos->data(),
                                                     sin->data(),
                                                     output->data(),
                                                     head_dim);
            }
            CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}