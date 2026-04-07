#include "base/tensor.h"
#include "base/cuda_library_context.h"
#include "base/util.h"
#include "kernel/kernel.h"
#include "kernel/cuda/mat_mul_backend_selector.h"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/gemm_kernel.cuh"
#include <cub/block/block_reduce.cuh>
#include <mutex>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        namespace
        {
            base::CudaLibraryContext &library_context()
            {
                thread_local base::CudaLibraryContext context;
                return context;
            }

            bool cuda_library_available()
            {
                static bool available = false;
                static std::once_flag probe_once;

                std::call_once(probe_once, []() {
                    cublasHandle_t handle = nullptr;
                    available = (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS);
                    if (available)
                    {
                        const cublasStatus_t destroy_status = cublasDestroy(handle);
                        if (destroy_status != CUBLAS_STATUS_SUCCESS)
                        {
                            LOG(WARNING) << "cuBLAS availability probe could not release handle: "
                                         << base::cublas_status_string(destroy_status);
                            available = false;
                        }
                    }
                });

                return available;
            }
        }

        // 朴素版本
        namespace simple
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

                float sum = 0.0f;
                size_t row_offset = row * K;
                for (int32_t i = threadIdx.x; i < K; i += blockDim.x)
                {
                    sum += mat0_data[row_offset + i] * mat1_data[i * M + col];
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
                size_t row_offset = row * K;

                using BlockReduce = cub::BlockReduce<float, 128>;

                for (uint32_t col = 0; col < M; col++)
                {
                    float sum = 0.0f;
                    for (uint32_t i = threadIdx.x; i < K; i += blockDim.x)
                    {
                        sum += mat0_data[row_offset + i] * mat1_data[i * M + col];
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

            __global__ void mat_sc_mul_kernel_cuda(float *mat_data, float *scalar_data, float *output_data, uint32_t N, uint32_t M)
            {
                uint32_t mat_id = blockIdx.x;
                uint32_t row_id = blockIdx.y;
                float scalar = *scalar_data;

                mat_data += mat_id * N * M + row_id * M;
                output_data += mat_id * N * M + row_id * M;

                for (uint32_t i = threadIdx.x; i < M; i += blockDim.x)
                {
                    output_data[i] = mat_data[i] * scalar;
                }
            }
        }

        void mat_mul_kernel_cuda_cublas(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            record_last_mat_mul_backend_execution(MatMulBackendExecution::LibraryBacked);

            CHECK(input0 != nullptr);
            CHECK(input1 != nullptr);
            CHECK(output != nullptr);

            CHECK_EQ(input0->device(), base::Device::CUDA);
            CHECK_EQ(input1->device(), base::Device::CUDA);
            CHECK_EQ(output->device(), base::Device::CUDA);

            CHECK(input0->is_contiguous());
            CHECK(input1->is_contiguous());
            CHECK(output->is_contiguous());

            CHECK_EQ(input0->shape().size(), 2u);
            CHECK_EQ(input1->shape().size(), 2u);
            CHECK_EQ(output->shape().size(), 2u);
            CHECK_GT(input1->size(), 1u);

            const size_t lhs_rows = input0->shape(0);
            const size_t lhs_cols = input0->shape(1);
            const size_t rhs_rows = input1->shape(0);
            const size_t rhs_cols = input1->shape(1);

            CHECK_EQ(lhs_cols, rhs_rows);
            CHECK_EQ(output->shape(0), lhs_rows);
            CHECK_EQ(output->shape(1), rhs_cols);

            input0->contiguous();
            input1->contiguous();
            output->contiguous();

            auto &context = library_context();
            context.bind_stream(static_cast<cudaStream_t>(stream));

            const float alpha = 1.0f;
            const float beta = 0.0f;
            CHECK_CUBLAS_ERR(cublasSgemm(context.cublas_handle(),
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         static_cast<int>(rhs_cols),
                                         static_cast<int>(lhs_rows),
                                         static_cast<int>(lhs_cols),
                                         &alpha,
                                         input1->data(),
                                         static_cast<int>(rhs_cols),
                                         input0->data(),
                                         static_cast<int>(lhs_cols),
                                         &beta,
                                         output->data(),
                                         static_cast<int>(rhs_cols)));
        }

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
            for (uint32_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                output_data[i] = mat_data[i] * scalar;
            }
        }

        void mat_mul_kernel_cuda_library_first(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            if ((base::is_floating_point_dtype(input0->dtype()) && input0->dtype() != base::DType::FP32) ||
                (base::is_floating_point_dtype(input1->dtype()) && input1->dtype() != base::DType::FP32) ||
                output->dtype() != base::DType::FP32)
            {
                base::Tensor input0_fp32 = input0->dtype() == base::DType::FP32 ? input0->clone() : input0->astype(base::DType::FP32);
                base::Tensor input1_fp32 = input1->dtype() == base::DType::FP32 ? input1->clone() : input1->astype(base::DType::FP32);
                base::Tensor output_fp32(output->shape(), base::Device::CUDA, output->is_mutable(), static_cast<cudaStream_t>(stream), base::DType::FP32);
                mat_mul_kernel_cuda_library_first(&input0_fp32, &input1_fp32, &output_fp32, stream);
                *output = output_fp32.astype(output->dtype());
                return;
            }

            MatMulBackendSelectionOptions options;
            options.library_enabled = true;
            options.library_available = cuda_library_available();
            options.allow_batched_matmul = false;

            const auto backend = select_mat_mul_backend(*input0, *input1, *output, options);
            if (backend.uses_library_backend())
            {
                mat_mul_kernel_cuda_cublas(input0, input1, output, stream);
                return;
            }

            mat_mul_kernel_cuda_vec(input0, input1, output, stream);
        }

        // TODO: 优化列数小的矩阵
        void mat_mul_kernel_cuda_vec(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, void *stream)
        {
            record_last_mat_mul_backend_execution(MatMulBackendExecution::HandwrittenFallback);

            if (input1->size() == 1)
            {
                CHECK(input0->shape() == output->shape());
                input0->contiguous();
                input1->contiguous();
                output->contiguous();
                size_t N = input0->shape(-2);
                size_t M = output->shape(-1);
                size_t num_mats = input0->num_mats();

                // CHECK_CUDA_ERR(cudaDeviceSynchronize());
                dim3 grid(num_mats, N);
                if (!stream)
                    LOG(WARNING) << "Using mat_sc_mul_kernel_cuda with default stream.";

                // 如果内存不对齐float4，则不适用向量化存取
                if (!align_float4(input0) || !align_float4(input1) || !align_float4(output))
                    simple::mat_sc_mul_kernel_cuda<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                                        input1->operator[](0),
                                                                                                        output->data(), N, M);
                else
                    mat_sc_mul_kernel_cuda<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(),
                                                                                                input1->operator[](0),
                                                                                                output->data(), N, M);

                // CHECK_CUDA_ERR(cudaDeviceSynchronize());
                return;
            }

            gemm_kernel(input0, input1, output, stream);
            return;

            // CHECK(input0->num_mats() > 0);
            // CHECK(input0->num_mats() == input1->num_mats());
            // CHECK(input0->num_mats() == output->num_mats());
            // CHECK(input0->shape(-1) == input1->shape(-2));
            // CHECK(input0->shape(-2) == output->shape(-2));
            // CHECK(input1->shape(-1) == output->shape(-1));
            // input0->contiguous();
            // input1->contiguous();
            // output->contiguous();
            // uint32_t num_mats = input0->num_mats();
            // uint32_t N = input0->shape(-2);
            // uint32_t K = input0->shape(-1);
            // uint32_t M = input1->shape(-1);
            // VLOG(DEBUG) << "Matrix multiplication dimensions: " << num_mats << "x" << N << "x" << K << " and " << K << "x" << M;

            // constexpr uint32_t MAX_BLOCK = 4096;
            // if (num_mats * N * M > MAX_BLOCK)
            // {
            //     // 每个线程块处理output中的一行
            //     dim3 blockDim(128);
            //     dim3 gridDim(num_mats, N);
            //     if (stream == nullptr)
            //         LOG(WARNING) << "Using mat_mul_kernel_cuda_fp32_vec for large matrix multiplication. without stream.";

            //     if (!align_float4(input0) || !align_float4(input1) || !align_float4(output))
            //         simple::mat_mul_kernel_cuda_fp32_vec_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
            //             input0->data(), input1->data(), output->data(), N, K, M);
            //     else
            //         mat_mul_kernel_cuda_fp32_vec_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
            //             input0->data(), input1->data(), output->data(), N, K, M);
            // }
            // else
            // {
            //     // 每个线程块处理output中的一个元素
            //     dim3 blockDim(128);
            //     dim3 gridDim(num_mats, N, M);
            //     if (stream == nullptr)
            //         LOG(WARNING) << "Using mat_mul_kernel_cuda_fp32_vec_fine for small matrix multiplication. without stream.";

            //     if (!align_float4(input0) || !align_float4(input1) || !align_float4(output))
            //         simple::mat_mul_kernel_cuda_fp32_vec_fine_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
            //             input0->data(), input1->data(), output->data(), N, K, M);
            //     else
            //         mat_mul_kernel_cuda_fp32_vec_fine_router<<<gridDim, blockDim, 0, static_cast<cudaStream_t>(stream)>>>(
            //             input0->data(), input1->data(), output->data(), N, K, M);
            // }
            // // CHECK_CUDA_ERR(cudaDeviceSynchronize());

            // CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}
