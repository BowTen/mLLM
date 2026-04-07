#include "kernel/cpu/contiguous_kernel.h"
#include "base/allocator.h"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        __global__ void contiguous_kernel_cuda_fp32(
            const float *old_data,
            float *new_data,
            size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
            size_t batch_size_stride, size_t num_heads_stride, size_t seq_len_stride, size_t head_dim_stride,
            size_t total_size)
        {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = gridDim.x * blockDim.x;
            for (size_t i = tid; i < total_size; i += stride)
            {
                size_t id = i;
                size_t offset = 0;
                offset += (id % head_dim) * head_dim_stride;
                id /= head_dim;
                offset += (id % seq_len) * seq_len_stride;
                id /= seq_len;
                offset += (id % num_heads) * num_heads_stride;
                id /= num_heads;
                offset += (id % batch_size) * batch_size_stride;
                new_data[i] = old_data[offset];
            }
        }

        template <typename T>
        __global__ void contiguous_kernel_cuda_typed(
            const T *old_data,
            T *new_data,
            size_t batch_size, size_t num_heads, size_t seq_len, size_t head_dim,
            size_t batch_size_stride, size_t num_heads_stride, size_t seq_len_stride, size_t head_dim_stride,
            size_t total_size)
        {
            size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
            size_t stride = gridDim.x * blockDim.x;
            for (size_t i = tid; i < total_size; i += stride)
            {
                size_t id = i;
                size_t offset = 0;
                offset += (id % head_dim) * head_dim_stride;
                id /= head_dim;
                offset += (id % seq_len) * seq_len_stride;
                id /= seq_len;
                offset += (id % num_heads) * num_heads_stride;
                id /= num_heads;
                offset += (id % batch_size) * batch_size_stride;
                new_data[i] = old_data[offset];
            }
        }

        void contiguous_kernel_cuda(base::Tensor *input,
                                    void *stream)
        {
            auto buffer = input->buffer()->clone(true);

            auto shape = input->shape();
            const size_t element_size = input->element_size();
            size_t dim = input->shape().size();
            size_t batch_size = dim >= 4 ? input->shape(-4) : 1;
            size_t num_heads = dim >= 3 ? input->shape(-3) : 1;
            size_t seq_len = dim >= 2 ? input->shape(-2) : 1;
            size_t head_dim = dim >= 1 ? input->shape(-1) : 1;
            size_t batch_size_stride = dim >= 4 ? input->stride(-4) : 0;
            size_t num_heads_stride = dim >= 3 ? input->stride(-3) : 0;
            size_t seq_len_stride = dim >= 2 ? input->stride(-2) : 0;
            size_t head_dim_stride = dim >= 1 ? input->stride(-1) : 0;
            size_t total_logic_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            if (total_logic_size > input->size())
            {
                auto vec_buffer = std::dynamic_pointer_cast<base::VecBuffer>(input->buffer());
                CHECK(vec_buffer != nullptr) << "Buffer is not VecBuffer, cannot resize.";
                vec_buffer->resize(total_logic_size * element_size);
            }
            auto *new_data = input->raw_data();
            auto *old_data = buffer->data();

            if (stream == nullptr)
                LOG(WARNING) << "Contiguous kernel: use default stream";
            if (input->dtype() == base::DType::FP32)
            {
                contiguous_kernel_cuda_fp32<<<8, 128, 0, static_cast<cudaStream_t>(stream)>>>(
                    static_cast<const float *>(old_data),
                    static_cast<float *>(new_data),
                    batch_size, num_heads, seq_len, head_dim,
                    batch_size_stride, num_heads_stride, seq_len_stride, head_dim_stride,
                    total_logic_size);
            }
            else if (input->dtype() == base::DType::BF16)
            {
                contiguous_kernel_cuda_typed<uint16_t><<<8, 128, 0, static_cast<cudaStream_t>(stream)>>>(
                    static_cast<const uint16_t *>(old_data),
                    static_cast<uint16_t *>(new_data),
                    batch_size, num_heads, seq_len, head_dim,
                    batch_size_stride, num_heads_stride, seq_len_stride, head_dim_stride,
                    total_logic_size);
            }
            else if (input->dtype() == base::DType::U32)
            {
                contiguous_kernel_cuda_typed<uint32_t><<<8, 128, 0, static_cast<cudaStream_t>(stream)>>>(
                    static_cast<const uint32_t *>(old_data),
                    static_cast<uint32_t *>(new_data),
                    batch_size, num_heads, seq_len, head_dim,
                    batch_size_stride, num_heads_stride, seq_len_stride, head_dim_stride,
                    total_logic_size);
            }
            else
            {
                CHECK(false) << "Unsupported tensor dtype in CUDA contiguous kernel.";
            }

            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}
