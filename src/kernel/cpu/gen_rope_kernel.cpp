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
                                 [[maybe_unused]] void *stream)
        {
            if (inv_freq->dtype() != base::DType::FP32 || cos->dtype() != base::DType::FP32 || sin->dtype() != base::DType::FP32)
            {
                base::Tensor inv_freq_fp32 = inv_freq->astype(base::DType::FP32);
                base::Tensor cos_fp32(cos->shape(), base::Device::CPU, cos->is_mutable(), nullptr, base::DType::FP32);
                base::Tensor sin_fp32(sin->shape(), base::Device::CPU, sin->is_mutable(), nullptr, base::DType::FP32);
                gen_rope_kernel_cpu(&inv_freq_fp32, pos_start, pos_end, &cos_fp32, &sin_fp32, stream);
                *cos = cos_fp32.astype(cos->dtype());
                *sin = sin_fp32.astype(sin->dtype());
                return;
            }

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
