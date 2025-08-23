#include "kernel/cpu/softmax_kernel.h"
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

namespace mllm
{
    namespace kernel
    {
        __global__ void softmax_kernel_cuda(float *input, float *output, size_t N, size_t M)
        {
            size_t mat_id = blockIdx.x;
            size_t row_id = blockIdx.y;

            input += mat_id * N * M + row_id * M;
            output += mat_id * N * M + row_id * M;
            float4 *input_vec = reinterpret_cast<float4 *>(input);
            float4 *output_vec = reinterpret_cast<float4 *>(output);
            size_t vec_size = M / 4;
            size_t vec_end = vec_size * 4;

            float sum = 0.f;
            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                sum += exp(input_val.x) + exp(input_val.y) + exp(input_val.z) + exp(input_val.w);
            }
            for (size_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                sum += exp(input[i]);
            }

            using BlockReduce = cub::BlockReduce<float, 128>;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ float shared_sum;
            sum = BlockReduce(temp_storage).Sum(sum);

            if (threadIdx.x == 0)
                shared_sum = sum;
            __syncthreads();
            sum = shared_sum;

            for (size_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                float4 input_val = input_vec[i];
                output_vec[i] = make_float4(
                    exp(input_val.x) / sum, exp(input_val.y) / sum, exp(input_val.z) / sum, exp(input_val.w) / sum);
            }
            for (size_t i = threadIdx.x + vec_end; i < M; i += blockDim.x)
            {
                output[i] = exp(input[i]) / sum;
            }
        }

        void softmax_kernel_cuda(base::Tensor *input, base::Tensor *output, void *stream)
        {
            CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
            size_t num_mats = input->num_mats();
            size_t n = input->shape(-2);
            size_t m = input->shape(-1);
            dim3 grid(num_mats, n);
            VLOG(TRACE) << "softmax cuda: num_mats: " << num_mats << ", n: " << n << ", m: " << m;

            if (stream)
            {
                softmax_kernel_cuda<<<grid, 128, 0, static_cast<cudaStream_t>(stream)>>>(input->data(), output->data(), n, m);
            }
            else
            {
                softmax_kernel_cuda<<<grid, 128>>>(input->data(), output->data(), n, m);
            }
        }
    }
}