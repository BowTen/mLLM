#include "kernel/cpu/embedding_kernel.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace kernel
    {
        void emb_kernel_cpu(base::Tensor *input,
                            base::Tensor *weight,
                            base::Tensor *output,
                            size_t vocab_size,
                            size_t hidden_size,
                            [[maybe_unused]] void *stream)
        {
            // Simple CPU implementation of the embedding kernel
            const uint32_t *input_data = reinterpret_cast<const uint32_t *>(input->data());
            const float *weight_data = weight->data();
            float *output_data = output->data();

            auto allocator = base::HostAllocator::getInstance();
            size_t input_size = input->size();
            for (size_t i = 0; i < input_size; ++i)
            {
                size_t idx = input_data[i];
                if (idx >= vocab_size)
                {
                    throw std::out_of_range("Input index out of range");
                }
                allocator->memcpy(&output_data[i * hidden_size],
                                  &weight_data[idx * hidden_size],
                                  hidden_size * sizeof(float));
            }
        }
    }
}