#include "kernel/cpu/softmax_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void row_softmax_kernel_cpu(float *input, float *output, size_t length)
        {
            float sum = 0.f;
            for (size_t i = 0; i < length; i++)
            {
                sum += exp(input[i]);
            }
            for (size_t i = 0; i < length; i++)
            {
                output[i] = exp(input[i]) / sum;
            }
        }

        void softmax_kernel_cpu(base::Tensor *input, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            CHECK(input->shape() == output->shape()) << "Input and output shapes must be the same.";
            size_t num_mats = input->num_mats();
            size_t n = input->shape(-2);
            size_t m = input->shape(-1);
            for (size_t i = 0; i < num_mats; i++)
            {
                float *input_mat = input->mat(i);
                float *output_mat = output->mat(i);
                for (size_t j = 0; j < n; j++)
                    row_softmax_kernel_cpu(input_mat + j * m, output_mat + j * m, m);
            }
        }
    }
}