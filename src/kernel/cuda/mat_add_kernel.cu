#include "kernel/cuda/mat_add_kernel.cuh"
#include "base/buffer.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void mat_add_kernel_cuda_fp32(const float *input0_data,
                                                 const float *input1_data,
                                                 float *output_data,
                                                 size_t tensor_dim,
                                                 size_t *shape,
                                                 size_t *stride0,
                                                 size_t *stride1,
                                                 size_t *stride_out,
                                                 size_t total_size)
        {
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= total_size)
                return;

            size_t offset_0 = 0, offset_1 = 0, offset_out = 0;
            for (int i = tensor_dim - 1; i >= 0; --i)
            {
                size_t coord = idx % shape[i];
                idx /= shape[i];

                offset_0 += coord * stride0[i];
                offset_1 += coord * stride1[i];
                offset_out += coord * stride_out[i];
            }
            output_data[offset_out] = input0_data[offset_0] + input1_data[offset_1];
        }

        void mat_add_kernel_cuda(base::Tensor *input0,
                                 base::Tensor *input1,
                                 base::Tensor *output,
                                 void *stream)
        {

            auto shape0 = input0->shape();
            auto shape1 = input1->shape();
            auto output_shape = output->shape();

            // 检查形状是否匹配
            if (shape0 != shape1 || shape0 != output_shape)
            {
                throw std::invalid_argument("Tensor shapes are not compatible for addition");
            }
            size_t num_mats = input0->num_mats();
            if (num_mats == 0)
            {
                throw std::invalid_argument("Input tensors must have at least one matrix.");
            }

            float *input0_data = input0->data();
            float *input1_data = input1->data();
            float *output_data = output->data();
            size_t tensor_dim = shape0.size();
            size_t total_size = input0->size();

            base::Allocator *allocator = base::CudaAllocator::getInstance();
            size_t buffer_size = tensor_dim * sizeof(size_t);
            base::ArrBuffer shape(allocator, buffer_size);
            base::ArrBuffer stride0(allocator, buffer_size);
            base::ArrBuffer stride1(allocator, buffer_size);
            base::ArrBuffer stride_out(allocator, buffer_size);
            cudaMemcpy(shape.data(), shape0.data(), buffer_size, cudaMemcpyHostToDevice);
            cudaMemcpy(stride0.data(), input0->stride().data(), buffer_size, cudaMemcpyHostToDevice);
            cudaMemcpy(stride1.data(), input1->stride().data(), buffer_size, cudaMemcpyHostToDevice);
            cudaMemcpy(stride_out.data(), output->stride().data(), buffer_size, cudaMemcpyHostToDevice);

            dim3 block(256);
            dim3 grid((total_size + block.x - 1) / block.x);

            if (stream != nullptr)
            {
                mat_add_kernel_cuda_fp32<<<grid, block, 0, (cudaStream_t)stream>>>(
                    input0_data,
                    input1_data,
                    output_data,
                    tensor_dim,
                    static_cast<size_t *>(shape.data()),
                    static_cast<size_t *>(stride0.data()),
                    static_cast<size_t *>(stride1.data()),
                    static_cast<size_t *>(stride_out.data()),
                    total_size);
            }
            else
            {
                LOG(WARNING) << "CUDA stream is null, using default stream.";
                mat_add_kernel_cuda_fp32<<<grid, block>>>(
                    input0_data,
                    input1_data,
                    output_data,
                    tensor_dim,
                    static_cast<size_t *>(shape.data()),
                    static_cast<size_t *>(stride0.data()),
                    static_cast<size_t *>(stride1.data()),
                    static_cast<size_t *>(stride_out.data()),
                    total_size);
            }
        }
    }
}
