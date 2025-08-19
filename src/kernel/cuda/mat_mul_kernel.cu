#include "base/tensor.h"
#include <cub/block/block_reduce.cuh>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        __global__ void mat_mul_kernel_cuda_fp32_vec_fine(float *input0_data,
                                                          float *input1_data,
                                                          float *output_data,
                                                          uint32_t hidden_size)
        {
            uint32_t vec_size = hidden_size / 4; // 每个线程处理4个元素
            uint32_t row = blockIdx.x;
            uint32_t col = blockIdx.y;
            uint32_t m = gridDim.y;
            uint32_t thread_id = threadIdx.x;

            float4 *input0_vec = reinterpret_cast<float4 *>(input0_data + row * hidden_size);

            float sum = 0.f;
            for (int32_t i = thread_id; i < vec_size; i += blockDim.x)
            {
                float4 a = input0_vec[i];
                sum += a.x * input1_data[i * 4 * m + col] +
                       a.y * input1_data[(i * 4 + 1) * m + col] +
                       a.z * input1_data[(i * 4 + 2) * m + col] +
                       a.w * input1_data[(i * 4 + 3) * m + col];
            }
            uint32_t vec_end = vec_size * 4;
            for (int32_t i = vec_end + thread_id; i < hidden_size; i += blockDim.x)
            {
                sum += input0_data[row * hidden_size + i] * input1_data[i * m + col];
            }

            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            sum = BlockReduce(temp_storage).Sum(sum);
            if (thread_id == 0)
            {
                output_data[row * m + col] = sum;
            }
        }

        __global__ void mat_mul_kernel_cuda_fp32_vec(float *input0_data,
                                                     float *input1_data,
                                                     float *output_data,
                                                     uint32_t K,
                                                     uint32_t M)
        {
            uint32_t vec_size = K / 4; // 每个线程处理4个元素
            uint32_t vec_end = vec_size * 4;
            uint32_t row = blockIdx.x;
            float4 *input0_vec = reinterpret_cast<float4 *>(input0_data + row * K);

            for (uint32_t j = 0; j < M; j++)
            {
                float sum = 0.f;
                for (int32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
                {
                    float4 a = input0_vec[i];
                    sum += a.x * input1_data[i * 4 * M + j] +
                           a.y * input1_data[(i * 4 + 1) * M + j] +
                           a.z * input1_data[(i * 4 + 2) * M + j] +
                           a.w * input1_data[(i * 4 + 3) * M + j];
                }
                for (int32_t i = vec_end + threadIdx.x; i < K; i += blockDim.x)
                {
                    sum += input0_data[row * K + i] * input1_data[i * M + j];
                }

                using BlockReduce = cub::BlockReduce<float, 128>;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                sum = BlockReduce(temp_storage).Sum(sum);
                if (threadIdx.x == 0)
                {
                    output_data[row * M + j] = sum;
                }
            }
        }

        // TODO: 优化列数小的矩阵
        void mat_mul_kernel_cuda_vec(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            auto shape0 = input0->shape();
            auto shape1 = input1->shape();
            uint32_t N = shape0[0];
            uint32_t K = shape0[1];
            uint32_t M = shape1[1];

            constexpr uint32_t MAX_BLOCK = 4096;
            if (N * M > MAX_BLOCK)
            {
                if (stream == nullptr)
                {
                    VLOG(DEBUG) << "Using mat_mul_kernel_cuda_fp32_vec for large matrix multiplication. without stream.";
                    mat_mul_kernel_cuda_fp32_vec<<<N, dim3(128)>>>(
                        input0->data(), input1->data(), output->data(), K, M);
                    VLOG(DEBUG) << "Finished mat_mul_kernel_cuda_fp32_vec for large matrix multiplication. without stream.";
                }
                else
                {
                    mat_mul_kernel_cuda_fp32_vec<<<N, dim3(128), 0, static_cast<cudaStream_t>(stream)>>>(
                        input0->data(), input1->data(), output->data(), K, M);
                }
            }
            else
            {
                dim3 blockDim(128);
                dim3 gridDim(N, M);
                if (stream == nullptr)
                {
                    VLOG(DEBUG) << "Using mat_mul_kernel_cuda_fp32_vec_fine for small matrix multiplication. without stream.";
                    mat_mul_kernel_cuda_fp32_vec_fine<<<gridDim, blockDim>>>(
                        input0->data(), input1->data(), output->data(), K);
                    VLOG(DEBUG) << "Finished mat_mul_kernel_cuda_fp32_vec_fine for small matrix multiplication. without stream.";
                }
                else
                {
                    mat_mul_kernel_cuda_fp32_vec_fine<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
                        input0->data(), input1->data(), output->data(), K);
                }
            }
        }
    }
}