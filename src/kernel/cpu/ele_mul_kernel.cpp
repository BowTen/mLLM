#include "kernel/cpu/ele_mul_kernel.h"

namespace mllm
{
    namespace kernel
    {
        void ele_mul_kernel_cpu(base::Tensor *input0, base::Tensor *input1, base::Tensor *output, [[maybe_unused]] void *stream)
        {
            if (input0->dtype() != base::DType::FP32 || input1->dtype() != base::DType::FP32 || output->dtype() != base::DType::FP32)
            {
                base::Tensor input0_fp32 = input0->astype(base::DType::FP32);
                base::Tensor input1_fp32 = input1->astype(base::DType::FP32);
                base::Tensor output_fp32(output->shape(), base::Device::CPU, output->is_mutable(), nullptr, base::DType::FP32);
                ele_mul_kernel_cpu(&input0_fp32, &input1_fp32, &output_fp32, stream);
                *output = output_fp32.astype(output->dtype());
                return;
            }

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
