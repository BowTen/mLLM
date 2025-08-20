#include "kernel/cpu/contiguous_kernel.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void contiguous_kernel_cuda_fp32(float *old_data,
                                                    float *new_data,
                                                    const size_t *shape,
                                                    const size_t *stride,
                                                    size_t dim,
                                                    size_t total_size,
                                                    void *stream)
        {
            for (int32_t i = threadIdx.x; i < total_size; i += blockDim.x)
            {
                auto id = i;
                size_t offset = 0;
                for (size_t j = 0; j < dim; j++)
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
            auto new_data = input->data();
            auto old_data = static_cast<float *>(buffer->data());

            size_t dim = input->shape().size();
            size_t total_size = input->size();

            auto stride = input->stride();
            auto shape = input->shape();
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
                                                                                               total_size,
                                                                                               stream);
            }
            else
            {
                contiguous_kernel_cuda_fp32<<<1, 1024>>>(old_data,
                                                         new_data,
                                                         static_cast<size_t *>(shape_cuda.data()),
                                                         static_cast<size_t *>(stride_cuda.data()),
                                                         dim,
                                                         total_size,
                                                         stream);
            }
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }
    }
}
