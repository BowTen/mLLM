#include "kernel/cpu/add_kernel.h"
#include <stdexcept>
#include <armadillo>

namespace mllm
{
    namespace kernel
    {
        void add_kernel_cpu(base::Tensor *input0,
                            base::Tensor *input1,
                            base::Tensor *output,
                            [[maybe_unused]] void *stream)
        {
            if (input1->size() == 1)
            {
                std::swap(input0, input1);
            }

            // 检查输入张量的形状是否兼容
            auto shape0 = input0->shape();
            auto shape1 = input1->shape();
            auto output_shape = output->shape();

            // 简单的形状检查：要么形状完全相同，要么其中一个是标量
            size_t size0 = input0->size();
            size_t size1 = input1->size();
            size_t output_size = output->size();

            if (size0 != size1 && size0 != 1)
            {
                throw std::invalid_argument("Tensor shapes are not compatible for element-wise addition");
            }

            if (output_size != std::max(size0, size1))
            {
                throw std::invalid_argument("Output tensor size does not match expected size");
            }

            if (size0 == 1)
            {
                arma::fvec vec1(input1->data(), size1, false, true);
                arma::fvec out_vec(output->data(), output_size, false, true);
                out_vec = vec1 + input0->data()[0]; // 将标量加到每个元素上
            }
            else
            {
                // 否则进行逐元素相加
                arma::fvec vec0(input0->data(), size0, false, true);
                arma::fvec vec1(input1->data(), size1, false, true);
                arma::fvec out_vec(output->data(), output_size, false, true);
                out_vec = vec0 + vec1;
            }
        }
    }
}
