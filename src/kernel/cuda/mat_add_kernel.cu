#include "kernel/cuda/mat_add_kernel.cuh"
#include "base/buffer.h"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void mat_add_kernel_cuda_fp32(float *input0_data,
                                                 float *input1_data,
                                                 float *output_data,
                                                 size_t total_size)
        {
            size_t vec_size = total_size / 4;
            size_t vec_end = vec_size * 4;
            float4 *input0_vec = reinterpret_cast<float4 *>(input0_data);
            float4 *input1_vec = reinterpret_cast<float4 *>(input1_data);
            float4 *output_vec = reinterpret_cast<float4 *>(output_data);
            for (size_t i = threadIdx.x; i < vec_size; i++)
            {
                float4 val0 = input0_vec[i];
                float4 val1 = input1_vec[i];
                output_vec[i] = make_float4(val0.x + val1.x,
                                            val0.y + val1.y,
                                            val0.z + val1.z,
                                            val0.w + val1.w);
            }
            for (size_t i = vec_end + threadIdx.x; i < total_size; i += blockDim.x)
            {
                output_data[i] = input0_data[i] + input1_data[i];
            }
        }

        void mat_add_kernel_cuda(base::Tensor *input0,
                                 base::Tensor *input1,
                                 base::Tensor *output,
                                 void *stream)
        {
            CHECK(input0->shape() == input1->shape());
            CHECK(input0->shape() == output->shape());
            input0->contiguous(static_cast<cudaStream_t>(stream));
            input1->contiguous(static_cast<cudaStream_t>(stream));
            output->contiguous(static_cast<cudaStream_t>(stream));

            dim3 block(1024);
            dim3 grid(1);

            if (stream != nullptr)
            {
                mat_add_kernel_cuda_fp32<<<grid, block, 0, (cudaStream_t)stream>>>(
                    input0->data(),
                    input1->data(),
                    output->data(),
                    input0->size());
            }
            else
            {
                LOG(WARNING) << "CUDA stream is null, using default stream.";
                mat_add_kernel_cuda_fp32<<<grid, block>>>(
                    input0->data(),
                    input1->data(),
                    output->data(),
                    input0->size());
            }
            CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}
