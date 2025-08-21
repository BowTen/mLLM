#include "base/tensor.h"
#include <cub/block/block_reduce.cuh>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        __device__ void row_rms_norm_kernel_cuda_fp32(float *input_data,
                                                      float *weight_data,
                                                      float *output_data,
                                                      size_t hidden_size,
                                                      float eps)
        {
            size_t vec_size = hidden_size / 4;
            size_t vec_end = vec_size * 4;

            float4 *input_vec = reinterpret_cast<float4 *>(input_data);
            float4 *weight_vec = reinterpret_cast<float4 *>(weight_data);
            float4 *output_vec = reinterpret_cast<float4 *>(output_data);

            float sum = 0.0f;
            for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                sum += input_val.x * input_val.x +
                       input_val.y * input_val.y +
                       input_val.z * input_val.z +
                       input_val.w * input_val.w;
            }
            for (uint32_t i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x)
            {
                sum += input_data[i] * input_data[i];
            }

            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float shared_sum;
            sum = BlockReduce(temp_storage).Sum(sum);
            if (threadIdx.x == 0)
            {
                shared_sum = sum;
            }
            __syncthreads();
            sum = shared_sum;

            float rsqrt = rsqrtf(sum / hidden_size + eps);
            for (int i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                float4 weight_val = weight_vec[i];
                output_vec[i].x = input_val.x * weight_val.x * rsqrt;
                output_vec[i].y = input_val.y * weight_val.y * rsqrt;
                output_vec[i].z = input_val.z * weight_val.z * rsqrt;
                output_vec[i].w = input_val.w * weight_val.w * rsqrt;
            }
            for (int i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x)
            {
                output_data[i] = input_data[i] * weight_data[i] * rsqrt;
            }
        }
        __global__ void rms_norm_kernel_cuda_fp32(float *input_data,
                                                  float *weight_data,
                                                  float *output_data,
                                                  size_t seq_size,
                                                  size_t hidden_size,
                                                  float eps)
        {
            size_t mat = blockIdx.x;
            size_t row = blockIdx.y;

            row_rms_norm_kernel_cuda_fp32(input_data + mat * hidden_size * seq_size + row * hidden_size,
                                          weight_data,
                                          output_data + mat * hidden_size * seq_size + row * hidden_size,
                                          hidden_size, eps);
        }

        void rms_norm_kernel_cuda(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, void *stream)
        {
            CHECK(input->shape() == output->shape());
            input->contiguous();
            weight->contiguous();
            output->contiguous();
            size_t seq_size = input->shape(-2);
            size_t hidden_size = weight->shape(-1);
            float *input_data = input->data();
            float *weight_data = weight->data();
            float *output_data = output->data();

            dim3 grid(input->num_mats(), seq_size);
            if (stream)
            {
                rms_norm_kernel_cuda_fp32<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input_data, weight_data, output_data, seq_size, hidden_size, eps);
            }
            else
            {
                LOG(WARNING) << "RMSNorm Kernel: Stream is null, using default stream";
                rms_norm_kernel_cuda_fp32<<<grid, 128>>>(input_data, weight_data, output_data, seq_size, hidden_size, eps);
            }
        }
    }
}