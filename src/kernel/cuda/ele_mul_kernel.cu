#include "kernel/cuda/ele_mul_kernel.cuh"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void ele_mul_kernel_cuda_fp32(float *input0,
                                                 float *input1,
                                                 float *output,
                                                 size_t total_size,
                                                 [[maybe_unused]] void *stream)
        {
            size_t vec_size = total_size / 4;
            size_t vec_end = vec_size * 4;
            float4 *vec_in0 = reinterpret_cast<float4 *>(input0);
            float4 *vec_in1 = reinterpret_cast<float4 *>(input1);
            float4 *vec_out = reinterpret_cast<float4 *>(output);
            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 in0_val = vec_in0[i];
                float4 in1_val = vec_in1[i];
                vec_out[i] = make_float4(
                    in0_val.x * in1_val.x,
                    in0_val.y * in1_val.y,
                    in0_val.z * in1_val.z,
                    in0_val.w * in1_val.w);
            }
            for (size_t i = vec_end + threadIdx.x; i < total_size; i += blockDim.x)
            {
                output[i] = input0[i] * input1[i];
            }
        }

        void ele_mul_kernel_cuda(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            CHECK(input0->shape() == input1->shape());
            input0->contiguous();
            input1->contiguous();
            output->contiguous();

            size_t total_size = input0->size();
            if (stream)
            {
                ele_mul_kernel_cuda_fp32<<<8, 128, 0, static_cast<cudaStream_t>(stream)>>>(input0->data(), input1->data(), output->data(), total_size, stream);
            }
            else
            {
                LOG(WARNING) << "No stream provided for CUDA ele mul kernel, running in default stream.";
                ele_mul_kernel_cuda_fp32<<<8, 128>>>(input0->data(), input1->data(), output->data(), total_size, stream);
            }
            CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}