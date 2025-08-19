#include "base/tensor.h"
#include <cub/block/block_reduce.cuh>

namespace mllm
{
    namespace kernel
    {
        __global__ void rms_norm_kernel_cuda_fp32(const float *input_data,
                                                  const float *weight_data,
                                                  float *output_data,
                                                  size_t hidden_size,
                                                  float eps)
        {
            const int bid = blockIdx.x;
            const int tid = threadIdx.x;

            float sum = 0.0f;
            for (int i = tid; i < hidden_size; i += blockDim.x)
            {
                sum += input_data[bid * hidden_size + i] * input_data[bid * hidden_size + i];
            }
            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float shared_sum;
            sum = BlockReduce(temp_storage).Sum(sum);
            if (tid == 0)
            {
                shared_sum = sum;
            }
            __syncthreads();
            sum = shared_sum;
            float rsqrt = rsqrtf(sum / hidden_size + eps);
            for (int i = tid; i < hidden_size; i += blockDim.x)
            {
                output_data[bid * hidden_size + i] = input_data[bid * hidden_size + i] * rsqrt * weight_data[i];
            }
        }
        __global__ void rms_norm_kernel_cuda_fp32_vec(const float *input_data,
                                                      const float *weight_data,
                                                      float *output_data,
                                                      size_t hidden_size,
                                                      float eps)
        {
            const int bid = blockIdx.x;
            const int tid = threadIdx.x;

            const float4 *input_vec = (const float4 *)(input_data);
            const float4 *weight_vec = (const float4 *)(weight_data);
            float4 *output_vec = (float4 *)(output_data);
            size_t vec_size = hidden_size / 4;

            float sum = 0.0f;
            for (int i = tid; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[bid * vec_size + i];
                sum += input_val.x * input_val.x + input_val.y * input_val.y + input_val.z * input_val.z + input_val.w * input_val.w;
            }
            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float shared_sum;
            sum = BlockReduce(temp_storage).Sum(sum);
            if (tid == 0)
            {
                shared_sum = sum;
            }
            __syncthreads();
            sum = shared_sum;
            float rsqrt = rsqrtf(sum / hidden_size + eps);
            for (int i = tid; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[bid * vec_size + i];
                float4 weight_val = weight_vec[i];
                output_vec[bid * vec_size + i] = make_float4(
                    input_val.x * rsqrt * weight_val.x,
                    input_val.y * rsqrt * weight_val.y,
                    input_val.z * rsqrt * weight_val.z,
                    input_val.w * rsqrt * weight_val.w);
            }
        }

        void rms_norm_kernel_cuda(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream)
        {
            size_t sqe_size = input->shape()[0];
            size_t hidden_size = weight->shape()[0];
            float *input_data = input->data();
            float *weight_data = weight->data();
            float *output_data = output->data();

            rms_norm_kernel_cuda_fp32<<<sqe_size, 128, 0, static_cast<cudaStream_t>(stream)>>>(input_data, weight_data, output_data, hidden_size, eps);
        }
        void rms_norm_kernel_cuda_vec(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream)
        {
            size_t sqe_size = input->shape()[0];
            size_t hidden_size = weight->shape()[0];
            CHECK(hidden_size % 4 == 0) << "Hidden size must be a multiple of 4 for vectorized RMS norm kernel.";
            float *input_data = input->data();
            float *weight_data = weight->data();
            float *output_data = output->data();

            rms_norm_kernel_cuda_fp32_vec<<<sqe_size, 128, 0, static_cast<cudaStream_t>(stream)>>>(input_data, weight_data, output_data, hidden_size, eps);
        }
    }
}