#include "kernel/cpu/contiguous_kernel.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void contiguous_kernel_cuda_fp32(float *old_data,
                                                    float *new_data,
                                                    const size_t *shape,
                                                    const size_t *stride,
                                                    int dim,
                                                    size_t total_size,
                                                    void *stream)
        {
            for (int32_t i = threadIdx.x; i < total_size; i += blockDim.x)
            {
                auto id = i;
                size_t offset = 0;
                for (int j = dim - 1; j >= 0; j--)
                {
                    offset += (id % shape[j]) * stride[j];
                    id /= shape[j];
                }
                new_data[i] = old_data[offset];
            }
        }

        void contiguous_kernel_cuda(base::Tensor *input,
                                    void *stream)
        {
            auto buffer = input->buffer()->clone(true);

            auto stride = input->stride();
            auto shape = input->shape();
            size_t dim = input->shape().size();
            size_t total_logic_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            if (total_logic_size > input->size())
            {
                auto vec_buffer = std::dynamic_pointer_cast<base::VecBuffer>(input->buffer());
                vec_buffer->resize(total_logic_size * sizeof(float));
            }
            auto new_data = input->data();
            auto old_data = static_cast<float *>(buffer->data());

            base::ArrBuffer stride_cuda(base::CudaAllocator::getInstance(), stride.size() * sizeof(size_t));
            base::ArrBuffer shape_cuda(base::CudaAllocator::getInstance(), shape.size() * sizeof(size_t));
            cudaMemcpy(stride_cuda.data(), stride.data(), stride.size() * sizeof(size_t), cudaMemcpyHostToDevice);
            cudaMemcpy(shape_cuda.data(), shape.data(), shape.size() * sizeof(size_t), cudaMemcpyHostToDevice);

            if (stream)
            {
                contiguous_kernel_cuda_fp32<<<1, 1024, 0, static_cast<cudaStream_t>(stream)>>>(old_data,
                                                                                               new_data,
                                                                                               static_cast<size_t *>(shape_cuda.data()),
                                                                                               static_cast<size_t *>(stride_cuda.data()),
                                                                                               dim,
                                                                                               total_logic_size,
                                                                                               stream);
            }
            else
            {
                contiguous_kernel_cuda_fp32<<<1, 1024>>>(old_data,
                                                         new_data,
                                                         static_cast<size_t *>(shape_cuda.data()),
                                                         static_cast<size_t *>(stride_cuda.data()),
                                                         dim,
                                                         total_logic_size,
                                                         stream);
            }
        }
    }
}
