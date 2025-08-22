#include "kernel/cpu/rope_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void row_rope_kernel_cpu(float *input,
                                 float *cos,
                                 float *sin,
                                 float *output,
                                 size_t head_dim)
        {
            size_t stride = head_dim / 2;
            for (size_t i = 0; i < stride; ++i)
            {
                size_t j = i + stride;
                float a = input[i] * cos[i] - input[j] * sin[i];
                float b = input[i] * sin[i] + input[j] * cos[i];
                output[i] = a;
                output[j] = b;
            }
        }

        void rope_kernel_cpu(base::Tensor *input,
                             base::Tensor *cos,
                             base::Tensor *sin,
                             base::Tensor *output,
                             void *stream)
        {
            CHECK(input->shape(-1) % 2 == 0) << "head dim must be even";
            CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
            CHECK(cos->shape() == sin->shape()) << "Cosine and sine shapes must be the same.";
            CHECK(input->shape(-1) == cos->shape(-1) && input->shape(-2) == cos->shape(-2));
            input->contiguous();
            cos->contiguous();
            sin->contiguous();
            output->contiguous();
            size_t head_dim = cos->shape(-1);
            size_t num_heads = input->num_mats();
            size_t seq_len = input->shape(-2);
            for (size_t i = 0; i < num_heads; i++)
            {
                for (size_t j = 0; j < seq_len; j++)
                {
                    row_rope_kernel_cpu(input->mat(i) + j * head_dim,
                                        cos->data() + j * head_dim,
                                        sin->data() + j * head_dim,
                                        output->mat(i) + j * head_dim,
                                        head_dim);
                }
            }
        }
    }
}