#include "kernel/cpu/silu_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void row_silu_kernel_cpu(float *input, size_t head_dim)
        {
            for (size_t i = 0; i < head_dim; i++)
            {
                input[i] = input[i] / (1 + exp(-input[i]));
            }
        }

        void silu_kernel_cpu(base::Tensor *input, [[maybe_unused]] void *stream)
        {
            size_t seq_len = input->shape(-2);
            input->contiguous();
            size_t head_dim = input->shape(-1);
            size_t num_mats = input->num_mats();
            for (size_t i = 0; i < num_mats; i++)
            {
                float *mat_data = input->mat(i);
                for (size_t j = 0; j < seq_len; j++)
                {
                    row_silu_kernel_cpu(mat_data + j * head_dim, head_dim);
                }
            }
        }
    }
}