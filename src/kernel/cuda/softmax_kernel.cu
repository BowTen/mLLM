#include "kernel/cuda/softmax_kernel.cuh"

#include <cfloat>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <mutex>

#include "base/cuda_library_context.h"
#include "base/util.h"
#include "kernel/cuda/softmax_backend_selector.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace kernel
    {
        namespace
        {
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            base::CudaLibraryContext &library_context()
            {
                thread_local base::CudaLibraryContext context;
                return context;
            }
#endif

            void ensure_handwritten_softmax_layout(base::Tensor *input, base::Tensor *output)
            {
                CHECK(input != nullptr);
                CHECK(output != nullptr);
                CHECK_EQ(input->device(), base::Device::CUDA);
                CHECK_EQ(output->device(), base::Device::CUDA);
                CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";

                if (!input->is_contiguous())
                {
                    input->contiguous();
                }
                if (!output->is_contiguous())
                {
                    output->contiguous();
                }
            }

            bool cuda_library_available()
            {
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
                static bool available = false;
                static std::once_flag probe_once;

                std::call_once(probe_once, []() {
                    cudnnHandle_t handle = nullptr;
                    available = (cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS);
                    if (available)
                    {
                        const cudnnStatus_t destroy_status = cudnnDestroy(handle);
                        if (destroy_status != CUDNN_STATUS_SUCCESS)
                        {
                            LOG(WARNING) << "cuDNN availability probe could not release handle: "
                                         << base::cudnn_status_string(destroy_status);
                            available = false;
                        }
                    }
                });

                return available;
#else
                return false;
#endif
            }

            __device__ float block_reduce_max(float value)
            {
                using BlockReduce = cub::BlockReduce<float, 128>;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                __shared__ float shared_value;

                const float reduced_value = BlockReduce(temp_storage).Reduce(value, cub::Max());
                if (threadIdx.x == 0)
                {
                    shared_value = reduced_value;
                }
                __syncthreads();
                return shared_value;
            }

            __device__ float block_reduce_sum(float value)
            {
                using BlockReduce = cub::BlockReduce<float, 128>;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                __shared__ float shared_value;

                const float reduced_value = BlockReduce(temp_storage).Sum(value);
                if (threadIdx.x == 0)
                {
                    shared_value = reduced_value;
                }
                __syncthreads();
                return shared_value;
            }

#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            void softmax_kernel_cuda_cudnn(base::Tensor *input, base::Tensor *output, void *stream)
            {
                record_last_softmax_backend_execution(SoftmaxBackendExecution::LibraryBacked);

                CHECK(input != nullptr);
                CHECK(output != nullptr);
                CHECK_EQ(input->device(), base::Device::CUDA);
                CHECK_EQ(output->device(), base::Device::CUDA);
                CHECK(input->is_contiguous());
                CHECK(output->is_contiguous());
                CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
                CHECK(input->shape().size() == 2u || input->shape().size() == 3u);

                input->contiguous();
                output->contiguous();

                const int rows = static_cast<int>(input->num_mats() * input->shape(-2));
                const int cols = static_cast<int>(input->shape(-1));

                auto &context = library_context();
                context.bind_stream(static_cast<cudaStream_t>(stream));

                cudnnTensorDescriptor_t descriptor = nullptr;
                CHECK_CUDNN_ERR(cudnnCreateTensorDescriptor(&descriptor));

                const float alpha = 1.0f;
                const float beta = 0.0f;

                CHECK_CUDNN_ERR(cudnnSetTensor4dDescriptor(descriptor,
                                                           CUDNN_TENSOR_NCHW,
                                                           CUDNN_DATA_FLOAT,
                                                           rows,
                                                           1,
                                                           1,
                                                           cols));
                CHECK_CUDNN_ERR(cudnnSoftmaxForward(context.cudnn_handle(),
                                                    CUDNN_SOFTMAX_ACCURATE,
                                                    CUDNN_SOFTMAX_MODE_INSTANCE,
                                                    &alpha,
                                                    descriptor,
                                                    input->data(),
                                                    &beta,
                                                    descriptor,
                                                    output->data()));
                CHECK_CUDNN_ERR(cudnnDestroyTensorDescriptor(descriptor));
            }
#endif
        } // namespace

        namespace simple
        {
            __global__ void softmax_kernel_cuda(float *input, float *output, size_t N, size_t M)
            {
                size_t mat_id = blockIdx.x;
                size_t row_id = blockIdx.y;

                input += mat_id * N * M + row_id * M;
                output += mat_id * N * M + row_id * M;

                float row_max = -FLT_MAX;
                for (size_t i = threadIdx.x; i < M; i += blockDim.x)
                {
                    row_max = fmaxf(row_max, input[i]);
                }
                row_max = block_reduce_max(row_max);

                float sum = 0.f;
                for (size_t i = threadIdx.x; i < M; i += blockDim.x)
                {
                    sum += expf(input[i] - row_max);
                }
                sum = block_reduce_sum(sum);

                for (size_t i = threadIdx.x; i < M; i += blockDim.x)
                {
                    output[i] = expf(input[i] - row_max) / sum;
                }
            }
        } // namespace simple

        __global__ void softmax_kernel_cuda_vectorized(float *input, float *output, size_t N, size_t M)
        {
            size_t mat_id = blockIdx.x;
            size_t row_id = blockIdx.y;

            input += mat_id * N * M + row_id * M;
            output += mat_id * N * M + row_id * M;
            float4 *input_vec = reinterpret_cast<float4 *>(input);
            float4 *output_vec = reinterpret_cast<float4 *>(output);
            size_t vec_size = M / 4;
            size_t vec_end = vec_size * 4;

            float row_max = -FLT_MAX;
            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                row_max = fmaxf(row_max, input_val.x);
                row_max = fmaxf(row_max, input_val.y);
                row_max = fmaxf(row_max, input_val.z);
                row_max = fmaxf(row_max, input_val.w);
            }
            for (size_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                row_max = fmaxf(row_max, input[i]);
            }
            row_max = block_reduce_max(row_max);

            float sum = 0.f;
            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                sum += expf(input_val.x - row_max) + expf(input_val.y - row_max) +
                       expf(input_val.z - row_max) + expf(input_val.w - row_max);
            }
            for (size_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                sum += expf(input[i] - row_max);
            }
            sum = block_reduce_sum(sum);

            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                output_vec[i] = make_float4(
                    expf(input_val.x - row_max) / sum,
                    expf(input_val.y - row_max) / sum,
                    expf(input_val.z - row_max) / sum,
                    expf(input_val.w - row_max) / sum);
            }
            for (size_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                output[i] = expf(input[i] - row_max) / sum;
            }
        }

        bool cuda_softmax_library_available()
        {
            return cuda_library_available();
        }

        void softmax_kernel_cuda_handwritten(base::Tensor *input, base::Tensor *output, void *stream)
        {
            record_last_softmax_backend_execution(SoftmaxBackendExecution::HandwrittenFallback);

            CHECK_EQ(input->device(), base::Device::CUDA);
            CHECK_EQ(output->device(), base::Device::CUDA);
            CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
            CHECK(input->is_contiguous()) << "Handwritten CUDA softmax requires contiguous input.";
            CHECK(output->is_contiguous()) << "Handwritten CUDA softmax requires contiguous output.";
            size_t num_mats = input->num_mats();
            size_t n = input->shape(-2);
            size_t m = input->shape(-1);
            dim3 grid(num_mats, n);

            if (stream == nullptr)
            {
                LOG(WARNING) << "Softmax: use default stream";
            }

            if (!align_float4(input) || !align_float4(output))
            {
                simple::softmax_kernel_cuda<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(), output->data(), n, m);
            }
            else
            {
                softmax_kernel_cuda_vectorized<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(), output->data(), n, m);
            }

            CHECK_CUDA_ERR(cudaGetLastError());
        }

        void softmax_kernel_cuda_library_first(base::Tensor *input, base::Tensor *output, void *stream)
        {
            if (input->dtype() != base::DType::FP32 || output->dtype() != base::DType::FP32)
            {
                base::Tensor input_fp32 = input->astype(base::DType::FP32);
                base::Tensor output_fp32(output->shape(), base::Device::CUDA, output->is_mutable(), static_cast<cudaStream_t>(stream), base::DType::FP32);
                softmax_kernel_cuda_library_first(&input_fp32, &output_fp32, stream);
                *output = output_fp32.astype(output->dtype());
                return;
            }

            SoftmaxBackendSelectionOptions options;
            options.library_enabled = true;
            options.library_available = cuda_library_available();

            const auto backend = select_softmax_backend(*input, *output, options);
            if (backend.uses_library_backend())
            {
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
                softmax_kernel_cuda_cudnn(input, output, stream);
                return;
#endif
            }

            ensure_handwritten_softmax_layout(input, output);
            softmax_kernel_cuda_handwritten(input, output, stream);
        }
    } // namespace kernel
} // namespace mllm
