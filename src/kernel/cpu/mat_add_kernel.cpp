#include "kernel/cpu/mat_add_kernel.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        void single_mat_add_kernel_cpu(float *mat0, float *mat1, float *out,
                                       size_t row, size_t col,
                                       size_t sr0, size_t sc0,
                                       size_t sr1, size_t sc1,
                                       size_t srout, size_t scout)
        {
            for (size_t i = 0; i < row; ++i)
            {
                for (size_t j = 0; j < col; ++j)
                {
                    out[i * srout + j * scout] = mat0[i * sr0 + j * sc0] + mat1[i * sr1 + j * sc1];
                }
            }
        }

        void mat_add_kernel_cpu(base::Tensor *input0,
                                base::Tensor *input1,
                                base::Tensor *output,
                                [[maybe_unused]] void *stream)
        {
            auto shape0 = input0->shape();
            auto shape1 = input1->shape();
            auto output_shape = output->shape();

            // 检查形状是否匹配
            if (shape0 != shape1 || shape0 != output_shape)
            {
                throw std::invalid_argument("Tensor shapes are not compatible for addition");
            }
            VLOG(TRACE) << "Running mat_add_kernel_cpu";
            size_t num_mats = input0->num_mats();
            VLOG(TRACE) << "Running mat_add_kernel_cpu with " << num_mats << " mats";
            if (num_mats == 0)
            {
                throw std::invalid_argument("Input tensors must have at least one matrix.");
            }

            for (size_t i = 0; i < num_mats; i++)
            {
                single_mat_add_kernel_cpu(input0->mat(i), input1->mat(i), output->mat(i),
                                          input0->shape(-2), input0->shape(-1),
                                          input0->stride(-2), input0->stride(-1),
                                          input1->stride(-2), input1->stride(-1),
                                          output->stride(-2), output->stride(-1));
            }
        }
    }
}
