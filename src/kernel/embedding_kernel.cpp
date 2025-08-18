#include "kernel/embedding_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void emb_kernel_cpu(base::Tensor *input,
                            base::Tensor *weight,
                            base::Tensor *output,
                            size_t vocab_size,
                            size_t hidden_size,
                            void *stream)
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

        void emb_kernel_cuda(base::Tensor *input,
                             base::Tensor *weight,
                             base::Tensor *output,
                             size_t vocab_size,
                             size_t hidden_size,
                             void *stream)
        {
            // CUDA implementation of the embedding kernel
            // This is a placeholder implementation. Replace with actual CUDA kernel launch.
            // For now, we will just throw an error to indicate it's not implemented.
            throw std::runtime_error("CUDA embedding kernel not implemented");
        }
    }
}