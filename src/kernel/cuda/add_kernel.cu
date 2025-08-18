#include "kernel/cuda/add_kernel.cuh"
#include <stdexcept>
#include <algorithm>

namespace mllm
{
    namespace kernel
    {
        // 普通版本的CUDA kernel
        __global__ void add_kernel_cuda_fp32(const float *input0_data,
                                             const float *input1_data,
                                             float *output_data,
                                             size_t size0,
                                             size_t size1,
                                             size_t output_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

            if (idx < output_size)
            {
                if (size0 == size1)
                {
                    // 两个张量形状相同
                    output_data[idx] = input0_data[idx] + input1_data[idx];
                }
                else if (size0 == 1)
                {
                    // input0 是标量，广播到 input1 的形状
                    output_data[idx] = input0_data[0] + input1_data[idx];
                }
                else if (size1 == 1)
                {
                    // input1 是标量，广播到 input0 的形状
                    output_data[idx] = input0_data[idx] + input1_data[0];
                }
            }
        }

        void add_kernel_cuda(base::Tensor *input0,
                             base::Tensor *input1,
                             base::Tensor *output,
                             void *stream)
        {
            if (stream == nullptr)
            {
                throw std::runtime_error("CUDA stream is null.");
            }

            size_t size0 = input0->size();
            size_t size1 = input1->size();
            size_t output_size = output->size();

            // 检查输入张量的形状是否兼容
            if (size0 != size1 && size0 != 1 && size1 != 1)
            {
                throw std::invalid_argument("Tensor shapes are not compatible for element-wise addition");
            }

            if (output_size != std::max(size0, size1))
            {
                throw std::invalid_argument("Output tensor size does not match expected size");
            }

            const float *input0_data = input0->data();
            const float *input1_data = input1->data();
            float *output_data = output->data();

            // 计算 grid 和 block 大小
            dim3 block(256);
            dim3 grid((output_size + block.x - 1) / block.x);

            add_kernel_cuda_fp32<<<grid, block, 0, (cudaStream_t)stream>>>(
                input0_data,
                input1_data,
                output_data,
                size0,
                size1,
                output_size);
        }
    }
}
