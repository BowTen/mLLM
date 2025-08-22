#include "kernel/cpu/gen_rope_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void gen_rope_kernel_cpu_row(float *inv_freq,
                                     float *cos,
                                     float *sin,
                                     size_t pos,
                                     size_t head_dim)
        {
            for (uint32_t i = 0; i < head_dim; ++i)
            {
                float angle = inv_freq[i] * pos;
                cos[i] = std::cos(angle);
                sin[i] = std::sin(angle);
            }
        }

        void gen_rope_kernel_cpu(base::Tensor *inv_freq,
                                 uint32_t pos_start,
                                 uint32_t pos_end,
                                 base::Tensor *cos,
                                 base::Tensor *sin,
                                 [[may_unused]] void *stream)
        {
            CHECK(inv_freq->size() == cos->shape(-1));
            CHECK(cos->shape() == sin->shape());
            CHECK(pos_end - pos_start == cos->shape(-2));
            inv_freq->contiguous();
            cos->contiguous();
            sin->contiguous();
            size_t num_mats = cos->num_mats();
            size_t head_dim = cos->shape(-1);
            size_t seq_len = cos->shape(-2);

            for (size_t i = 0; i < num_mats; i++)
            {
                for (size_t j = 0; j < seq_len; j++)
                {
                    gen_rope_kernel_cpu_row(inv_freq->data(),
                                            cos->mat(i) + j * head_dim,
                                            sin->mat(i) + j * head_dim,
                                            j + pos_start,
                                            head_dim);
                }
            }
        }
    }
}