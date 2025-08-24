#include "kernel/cpu/ele_mul_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void ele_mul_kernel_cpu(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            CHECK(input0->shape() == input1->shape());
            input0->contiguous();
            input1->contiguous();
            output->contiguous();

            size_t total_size = input0->size();
            arma::fvec in0(input0->data(), total_size, false, true);
            arma::fvec in1(input1->data(), total_size, false, true);
            arma::fvec out(output->data(), total_size, false, true);
            out = in0 % in1;
        }
    }
}