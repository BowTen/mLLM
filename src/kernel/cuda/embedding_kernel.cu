#include "kernel/cuda/embedding_kernel.cuh"

namespace mllm
{
    namespace kernel
    {
        __global__ void emb_kernel_cuda_fp32(const uint32_t *input_data,
                                             const float *weight_data,
                                             float *output_data,
                                             uint32_t vocab_size,
                                             uint32_t hidden_size,
                                             uint32_t input_size)
        {
            size_t token = blockIdx.x;
            uint32_t token_id = input_data[token];
            for (uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x)
            {
                output_data[token * hidden_size + i] = weight_data[token_id * hidden_size + i];
            }
        }

        __global__ void emb_kernel_cuda_fp32_vec(const uint32_t *input_data,
                                                 const float *weight_data,
                                                 float *output_data,
                                                 uint32_t vocab_size,
                                                 uint32_t hidden_size,
                                                 uint32_t input_size)
        {
            size_t token = blockIdx.x;
            uint32_t token_id = input_data[token];
            float4 *weight_data_vec = (float4 *)weight_data;
            float4 *output_data_vec = (float4 *)output_data;
            uint32_t hidden_size_vec = hidden_size / 4;
            for (uint32_t i = threadIdx.x; i < hidden_size_vec; i += blockDim.x)
            {
                output_data_vec[token * hidden_size_vec + i] = weight_data_vec[token_id * hidden_size_vec + i];
            }
        }

        void emb_kernel_cuda(base::Tensor *input,
                             base::Tensor *weight,
                             base::Tensor *output,
                             size_t vocab_size,
                             size_t hidden_size,
                             void *stream)
        {
            CHECK(stream != nullptr) << "CUDA stream is null.";

            auto shape = input->shape();
            size_t seq_len = shape.size() == 2 ? shape[0] : 1;

            uint32_t *input_data = (uint32_t *)input->data();
            float *weight_data = (float *)weight->data();
            float *output_data = (float *)output->data();

            emb_kernel_cuda_fp32<<<seq_len, 128, 0, (cudaStream_t)stream>>>(
                input_data,
                weight_data,
                output_data,
                vocab_size,
                hidden_size,
                seq_len);
        }
        void emb_kernel_cuda_vec(base::Tensor *input,
                                 base::Tensor *weight,
                                 base::Tensor *output,
                                 size_t vocab_size,
                                 size_t hidden_size,
                                 void *stream)
        {
            CHECK(hidden_size % 4 == 0) << "hidden_size must be multiple of 4 for vectorized version.";
            CHECK(stream != nullptr) << "CUDA stream is null.";

            auto shape = input->shape();
            size_t seq_len = shape.size() == 2 ? shape[0] : 1;

            uint32_t *input_data = (uint32_t *)input->data();
            float *weight_data = (float *)weight->data();
            float *output_data = (float *)output->data();

            emb_kernel_cuda_fp32_vec<<<seq_len, 128, 0, (cudaStream_t)stream>>>(
                input_data,
                weight_data,
                output_data,
                vocab_size,
                hidden_size,
                seq_len);
        }
    }
}