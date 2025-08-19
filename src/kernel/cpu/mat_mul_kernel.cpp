#include "kernel/cpu/mat_mul_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void mat_mul_kernel_cpu(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            auto shape0 = input0->shape();
            auto shape1 = input1->shape();
            auto output_shape = output->shape();
            CHECK(shape0.size() == 2 && shape1.size() == 2 && output_shape.size() == 2) << "Input tensors must be 2D matrices";
            CHECK(shape0[1] == shape1[0]) << "Inner dimensions must match for matrix multiplication";
            CHECK(output_shape[0] == shape0[0] && output_shape[1] == shape1[1]) << "Output shape must match the result of matrix multiplication";

            arma::fmat mat0(input0->data(), shape0[1], shape0[0], false, true);
            arma::fmat mat1(input1->data(), shape1[1], shape1[0], false, true);
            arma::fmat mat_out(output->data(), output_shape[1], output_shape[0], false, true);

            mat_out = mat1 * mat0;
        }
    }
}