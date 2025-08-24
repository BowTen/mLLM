#include "kernel/cpu/causal_mask_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void causal_mask_kernel_cpu(base::Tensor *input,
                                    [[maybe_unused]] void *stream)
        {
            size_t seq_len = input->shape(-2);
            if (seq_len <= 1)
                return;
            input->contiguous();
            size_t head_dim = input->shape(-1);
            size_t num_mats = input->num_mats();
            for (size_t i = 0; i < num_mats; i++)
            {
                float *mat_data = input->mat(i);
                for (int j = seq_len - 2; j >= 0; j--)
                {
                    float *row_data = mat_data + j * head_dim;
                    for (size_t k = head_dim - (seq_len - j) + 1; k < head_dim; k++)
                    {
                        row_data[k] = -FLOAT_INF;
                    }
                }
            }
        }
    }
}