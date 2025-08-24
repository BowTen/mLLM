#include "kernel/cpu/contiguous_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void contiguous_kernel_cpu(base::Tensor *input,
                                   [[maybe_unused]] void *stream)
        {
            auto buffer = input->buffer()->clone(true);

            auto stride = input->stride();
            auto shape = input->shape();
            size_t total_logic_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            if (total_logic_size > input->size())
            {
                VLOG(DEBUG) << "Resize buffer to fit the Tensor logic size. logic: " << total_logic_size << ", size: " << input->size();
                auto vec_buffer = std::dynamic_pointer_cast<base::VecBuffer>(input->buffer());
                CHECK(vec_buffer != nullptr) << "Buffer is not VecBuffer, cannot resize.";
                vec_buffer->resize(total_logic_size * sizeof(float));
            }
            int dim = stride.size();
            auto new_data = input->data();
            auto old_data = static_cast<float *>(buffer->data());
            for (size_t i = 0; i < total_logic_size; i++)
            {
                size_t id = i;
                size_t offset = 0;
                for (int j = dim - 1; j >= 0; j--)
                {
                    offset += (id % shape[j]) * stride[j];
                    id /= shape[j];
                }
                new_data[i] = old_data[offset];
            }
        }
    }
}
