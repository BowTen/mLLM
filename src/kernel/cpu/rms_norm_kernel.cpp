#ifndef MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H
#define MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H

#include "base/tensor.h"
#include <armadillo>

namespace mllm
{
    namespace kernel
    {

        void rms_norm_kernel_cpu(base::Tensor *input, base::Tensor *weight, base::Tensor *output, float eps, [[maybe_unused]] void *stream)
        {
            auto shape = input->shape();
            uint32_t sqe_size = shape[0];
            uint32_t hidden_size = shape[1];
            float *input_data = input->data();
            float *weight_data = weight->data();
            float *output_data = output->data();

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
    }
}

#endif // MLLM_KERNEL_CPU_RMS_NORM_KERNEL_H