#include "kernel/cuda/gen_rope_kernel.cuh"

namespace mllm
{
    namespace kernel
    {
        __global__ void gen_rope_kernel_cpu(float *inv_freq,
                                            float *cos,
                                            float *sin,
                                            size_t pos_start,
                                            size_t head_dim)
        {
            size_t mat_id = blockIdx.x;
            size_t token_id = blockIdx.y;
            size_t pos = pos_start + token_id;
            size_t seq_len = gridDim.y;

            cos += mat_id * head_dim * seq_len + token_id * seq_len;
            sin += mat_id * head_dim * seq_len + token_id * seq_len;

            float4 *inv_freq_vec = reinterpret_cast<float4 *>(inv_freq);
            float4 *cos_vec = reinterpret_cast<float4 *>(cos);
            float4 *sin_vec = reinterpret_cast<float4 *>(sin);
            size_t vec_size = head_dim / 4;
            size_t vec_end = vec_size * 4;
            for (size_t i = threadIdx.x; i < vec_size; i++)
            {
                float4 inv_freq_val = inv_freq_vec[i];
                cos_vec[i] = make_float4(
                    cosf(pos * inv_freq_val.x),
                    cosf(pos * inv_freq_val.y),
                    cosf(pos * inv_freq_val.z),
                    cosf(pos * inv_freq_val.w));
                sin_vec[i] = make_float4(
                    sinf(pos * inv_freq_val.x),
                    sinf(pos * inv_freq_val.y),
                    sinf(pos * inv_freq_val.z),
                    sinf(pos * inv_freq_val.w));
            }
            for (size_t i = vec_end + threadIdx.x; i < head_dim; i += blockDim.x)
            {
                float inv_freq_val = inv_freq[i];
                cos[i] = cosf(pos * inv_freq_val);
                sin[i] = sinf(pos * inv_freq_val);
            }
        }

        void gen_rope_kernel_cuda(base::Tensor *inv_freq,
                                  uint32_t pos_start,
                                  uint32_t pos_end,
                                  base::Tensor *cos,
                                  base::Tensor *sin,
                                  void *stream)
        {
            CHECK(inv_freq->size() == cos->shape(-1));
            CHECK(cos->shape() == sin->shape());
            CHECK(pos_end - pos_start == cos->shape(-2));
            inv_freq->contiguous();
            cos->contiguous();
            sin->contiguous();
            size_t num_mats = cos->num_mats();
            size_t head_dim = cos->shape(-1);
            size_t seq_len = cos->shape(-2);

            dim3 grid(num_mats, seq_len);
            if (stream)
            {
                gen_rope_kernel_cpu<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
                    inv_freq->data(),
                    cos->data(),
                    sin->data(),
                    pos_start,
                    head_dim);
            }
            else
            {
                LOG(WARNING) << "Gen RoPE: use default cuda stream";
                gen_rope_kernel_cpu<<<grid, 256>>>(
                    inv_freq->data(),
                    cos->data(),
                    sin->data(),
                    pos_start,
                    head_dim);
            }
        }
    }
}