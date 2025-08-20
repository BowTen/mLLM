#include "kernel/cpu/contiguous_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void contiguous_kernel_cpu(base::Tensor *input,
                                   [[maybe_unused]] void *stream)
        {
            auto buffer = input->buffer()->clone(true);
            auto new_data = input->data();
            auto old_data = static_cast<float *>(buffer->data());

            auto stride = input->stride();
            auto shape = input->shape();
            size_t dim = stride.size();
            size_t total_size = input->size();
            for (size_t i = 0; i < total_size; i++)
            {
                size_t id = i;
                size_t offset = 0;
                for (size_t j = 0; j < dim; j++)
                {
                    offset += (id % shape[j]) * stride[j];
                    id /= shape[j];
                }
                new_data[i] = old_data[offset];
            }
        }
    }
}
