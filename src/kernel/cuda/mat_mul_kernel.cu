#include "base/tensor.h"
#include <cub/block/block_reduce.cuh>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        __device__ void mat_mul_kernel_cuda_fp32_vec_fine(float *mat0_data,
                                                          float *mat1_data,
                                                          float *output_data,
                                                          uint32_t N,
                                                          uint32_t K,
                                                          uint32_t M)
        {
            int32_t row = blockIdx.y;
            int32_t col = blockIdx.z;

            float4 *row_data_vec = reinterpret_cast<float4 *>(mat0_data + row * K);
            int32_t vec_size = K / 4; // 每个线程处理4个元素
            int32_t vec_end = vec_size * 4;

            float sum = 0.0f;
            for (int32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 a = row_data_vec[i];
                sum += a.x * mat1_data[i * 4 * M + col] +
                       a.y * mat1_data[(i * 4 + 1) * M + col] +
                       a.z * mat1_data[(i * 4 + 2) * M + col] +
                       a.w * mat1_data[(i * 4 + 3) * M + col];
            }
            for (int32_t i = vec_end + threadIdx.x; i < K; i += blockDim.x)
            {
                sum += mat0_data[row * K + i] * mat1_data[i * M + col];
            }

            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            sum = BlockReduce(temp_storage).Sum(sum);
            if (threadIdx.x == 0)
            {
                output_data[row * M + col] = sum;
            }
        }
        __global__ void mat_mul_kernel_cuda_fp32_vec_fine_router(float *input0_data,
                                                                 float *input1_data,
                                                                 float *output_data,
                                                                 uint32_t N,
                                                                 uint32_t K,
                                                                 uint32_t M)
        {
            uint32_t mat_id = blockIdx.x;
            mat_mul_kernel_cuda_fp32_vec_fine(input0_data + mat_id * N * K,
                                              input1_data + mat_id * K * M,
                                              output_data + mat_id * N * M,
                                              N, K, M);
        }

        __device__ void mat_mul_kernel_cuda_fp32_vec(float *mat0_data,
                                                     float *mat1_data,
                                                     float *output_data,
                                                     uint32_t N,
                                                     uint32_t K,
                                                     uint32_t M)
        {
            uint32_t row = blockIdx.y;

            float4 *row_data_vec = reinterpret_cast<float4 *>(mat0_data + row * K);
            uint32_t vec_size = K / 4; // 每个线程处理4个元素
            uint32_t vec_end = vec_size * 4;

            using BlockReduce = cub::BlockReduce<float, 128>;

            for (uint32_t col = 0; col < M; col++)
            {
                float sum = 0.0f;
                for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
                {
                    float4 a = row_data_vec[i];
                    sum += a.x * mat1_data[i * 4 * M + col] +
                           a.y * mat1_data[(i * 4 + 1) * M + col] +
                           a.z * mat1_data[(i * 4 + 2) * M + col] +
                           a.w * mat1_data[(i * 4 + 3) * M + col];
                }
                for (uint32_t i = vec_end + threadIdx.x; i < K; i += blockDim.x)
                {
                    sum += mat0_data[row * K + i] * mat1_data[i * M + col];
                }
                __shared__ typename BlockReduce::TempStorage temp_storage;
                sum = BlockReduce(temp_storage).Sum(sum);
                if (threadIdx.x == 0)
                {
                    output_data[row * M + col] = sum;
                }
                __syncthreads();
            }
        }
        __global__ void mat_mul_kernel_cuda_fp32_vec_router(float *input0_data,
                                                            float *input1_data,
                                                            float *output_data,
                                                            uint32_t N,
                                                            uint32_t K,
                                                            uint32_t M)
        {
            uint32_t mat_id = blockIdx.x;
            mat_mul_kernel_cuda_fp32_vec(input0_data + mat_id * N * K,
                                         input1_data + mat_id * K * M,
                                         output_data + mat_id * N * M,
                                         N, K, M);
        }

        void _mat_mul(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, uint32_t N, uint32_t K, uint32_t M)
        {
            using namespace std;
            input0->toDevice(base::Device::CPU);
            input1->toDevice(base::Device::CPU);
            output->toDevice(base::Device::CPU);
            cout << "input0: \n";
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < K; j++)
                {
                    cout << *((*input0)[std::vector<size_t>({i, j})]) << " ";
                }
                cout << '\n';
            }
            cout << "input1: \n";
            for (size_t i = 0; i < K; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    cout << *((*input1)[std::vector<size_t>({i, j})]) << " ";
                }
                cout << '\n';
            }
            cout << "output:\n";
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    *((*output)[std::vector<size_t>({i, j})]) = 0.0f;
                    for (size_t k = 0; k < K; k++)
                    {
                        *((*output)[std::vector<size_t>({i, j})]) +=
                            *((*input0)[std::vector<size_t>({i, k})]) * *((*input1)[std::vector<size_t>({k, j})]);
                    }
                    cout << *((*output)[std::vector<size_t>({i, j})]) << ' ';
                }
                cout << '\n';
            }
            // input0->toDevice(base::Device::CUDA);
            // input1->toDevice(base::Device::CUDA);
            // output->toDevice(base::Device::CUDA);
        }

        __global__ void mat_sc_mul_kernel_cuda(float *mat_data, float *scalar_data, float *output_data, uint32_t N, uint32_t M)
        {
            uint32_t mat_id = blockIdx.x;
            uint32_t row_id = blockIdx.y;
            float scalar = *scalar_data;

            mat_data += mat_id * N * M + row_id * M;
            output_data += mat_id * N * M + row_id * M;
            float4 *mat_vec = reinterpret_cast<float4 *>(mat_data);
            float4 *output_vec = reinterpret_cast<float4 *>(output_data);

            uint32_t vec_size = M / 4;
            uint32_t vec_end = vec_size * 4;
            for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 mat_val = mat_vec[i];
                output_vec[i] = make_float4(
                    mat_val.x * scalar,
                    mat_val.y * scalar,
                    mat_val.z * scalar,
                    mat_val.w * scalar);
            }
            for (uint32_t i = threadIdx.x + vec_end; i < M; i++)
            {
                output_data[i] = mat_data[i] * scalar;
            }
        }

        // TODO: 优化列数小的矩阵
        void mat_mul_kernel_cuda_vec(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            if (input1->size() == 1)
            {
                CHECK(input0->shape() == output->shape());
                input0->contiguous();
                input1->contiguous();
                output->contiguous();
                size_t N = input0->shape(-2);
                size_t M = output->shape(-1);
                size_t num_mats = input0->num_mats();

                dim3 grid(num_mats, N);
                if (stream)
                {
                    mat_sc_mul_kernel_cuda<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                                input1->operator[](0),
                                                                                                output->data(), N, M);
                }
                else
                {
                    LOG(WARNING) << "Using mat_sc_mul_kernel_cuda with default stream.";
                    mat_sc_mul_kernel_cuda<<<grid, 128>>>(input0->data(),
                                                          input1->operator[](0),
                                                          output->data(), N, M);
                }

                return;
            }

            CHECK(input0->num_mats() > 0);
            CHECK(input0->num_mats() == input1->num_mats());
            CHECK(input0->num_mats() == output->num_mats());
            CHECK(input0->shape(-1) == input1->shape(-2));
            CHECK(input0->shape(-2) == output->shape(-2));
            CHECK(input1->shape(-1) == output->shape(-1));
            input0->contiguous();
            input1->contiguous();
            output->contiguous();
            uint32_t num_mats = input0->num_mats();
            uint32_t N = input0->shape(-2);
            uint32_t K = input0->shape(-1);
            uint32_t M = input1->shape(-1);
            VLOG(DEBUG) << "Matrix multiplication dimensions: " << num_mats << "x" << N << "x" << K << " and " << K << "x" << M;

            constexpr uint32_t MAX_BLOCK = 4096;
            if (num_mats * N * M > MAX_BLOCK)
            {
                // 每个线程块处理output中的一行
                dim3 blockDim(128);
                dim3 gridDim(num_mats, N);
                if (stream == nullptr)
                {
                    LOG(WARNING) << "Using mat_mul_kernel_cuda_fp32_vec for large matrix multiplication. without stream.";
                    mat_mul_kernel_cuda_fp32_vec_router<<<gridDim, blockDim>>>(
                        input0->data(), input1->data(), output->data(), N, K, M);
                }
                else
                {
                    mat_mul_kernel_cuda_fp32_vec_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
                        input0->data(), input1->data(), output->data(), N, K, M);
                }
            }
            else
            {
                // 每个线程块处理output中的一个元素
                dim3 blockDim(128);
                dim3 gridDim(num_mats, N, M);
                if (stream == nullptr)
                {
                    LOG(WARNING) << "Using mat_mul_kernel_cuda_fp32_vec_fine for small matrix multiplication. without stream.";
                    mat_mul_kernel_cuda_fp32_vec_fine_router<<<gridDim, blockDim>>>(
                        input0->data(), input1->data(), output->data(), N, K, M);
                }
                else
                {
                    mat_mul_kernel_cuda_fp32_vec_fine_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
                        input0->data(), input1->data(), output->data(), N, K, M);
                }
            }
        }
    }
}