#include "kernel/cpu/random_sampling_kernel.h"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        void random_sampling_kernel_cpu(base::Tensor *probability,
                                        base::Tensor *output,
                                        [[maybe_unused]] void *stream)
        {
            float rand_num = base::get_random_float();

            float eps = 1e-6;
            size_t vocab_size = probability->shape(-1);
            float *prob_data = probability->data();
            for (uint32_t i = 0; i < vocab_size; i++)
            {
                rand_num -= prob_data[i];
                if (rand_num < eps)
                {
                    reinterpret_cast<uint32_t *>(output->data())[0] = i;
                    break;
                }
            }
            CHECK_LT(rand_num, eps);
        }
    }
}