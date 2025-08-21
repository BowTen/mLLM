#include "kernel/cpu/mat_mul_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void single_mat_mul_kernel_cpu(float *mat0_data, float *mat1_data, float *mat_out_data,
                                       uint32_t N, uint32_t K, uint32_t M)
        {
            arma::fmat mat0(mat0_data, K, N, false, true);
            arma::fmat mat1(mat1_data, M, K, false, true);
            arma::fmat mat_out(mat_out_data, M, N, false, true);

            mat_out = mat1 * mat0;
        }

        void mat_mul_kernel_cpu(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            CHECK(input0->num_mats() > 0);
            CHECK(input0->num_mats() == input1->num_mats());
            CHECK(input0->num_mats() == output->num_mats());
            CHECK(input0->shape(-1) == input1->shape(-2));
            CHECK(input0->shape(-2) == output->shape(-2));
            CHECK(input1->shape(-1) == output->shape(-1));
            input0->contiguous();
            input1->contiguous();
            output->contiguous();
            uint32_t num_mats = input0->num_mats();
            uint32_t N = input0->shape(-2);
            uint32_t K = input0->shape(-1);
            uint32_t M = input1->shape(-1);

            for (uint32_t i = 0; i < num_mats; i++)
            {
                single_mat_mul_kernel_cpu(input0->mat(i), input1->mat(i), output->mat(i), N, K, M);
            }
        }
    }
}