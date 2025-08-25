#ifndef MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H
#define MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H

#include "base/tensor.h"
#include <armadillo>

namespace mllm
{
    namespace kernel
    {
        void mat_rms_norm_kernel_cpu(
            float *input_data,
            float *weight_data,
            float *output_data,
            float eps,
            uint32_t sqe_size,
            uint32_t hidden_size)
        {
            arma::fvec vec_weight(weight_data, hidden_size, false, true);
            for (uint32_t i = 0; i < sqe_size; i++)
            {
                arma::fvec vec_input(input_data + i * hidden_size, hidden_size, false, true);
                arma::fvec vec_output(output_data + i * hidden_size, hidden_size, false, true);
                float mean = arma::as_scalar(arma::mean(arma::pow(vec_input, 2)));
                float rsqrt = 1.0f / std::sqrt(mean + eps);
                vec_output = (vec_input * rsqrt) % vec_weight;
            }
        }

        void rms_norm_kernel_cpu(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, [[maybe_unused]] void *stream)
        {
            CHECK(input->shape() == output->shape());
            CHECK(input->num_mats() > 0);
            CHECK(weight->shape(-1) == input->shape(-1));
            input->contiguous();
            weight->contiguous();
            output->contiguous();
            size_t num_mats = input->num_mats();
            for (size_t i = 0; i < num_mats; i++)
            {
                mat_rms_norm_kernel_cpu(
                    input->mat(i),
                    weight->data(),
                    output->mat(i),
                    eps,
                    input->shape(-2),
                    input->shape(-1));
            }
        }
    }
}

#endif // MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H