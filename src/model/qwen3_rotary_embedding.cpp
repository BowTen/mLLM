#include "model/qwen3.h"
#include "model/qwen3_rotary_embedding.h"
#include "base/util.h"
#include "kernel/kernel.h"

namespace mllm
{
    namespace model
    {
        Qwen3RotaryEmbedding::Qwen3RotaryEmbedding(op::JsonConfig config,
                                                   base::Device device,
                                                   cudaStream_t stream,
                                                   base::DType inference_dtype)
            : device_(device), stream_(stream), inference_dtype_(inference_dtype), head_dim(config["head_dim"]), rope_theta(config["rope_theta"])
        {
            std::vector<float> inv_freq_data(head_dim / 2);
            for (size_t i = 0; i < inv_freq_data.size(); i++)
                inv_freq_data[i] = 1.0f / std::pow(rope_theta, static_cast<float>(i) / (head_dim / 2));
            inv_freq_data.insert(inv_freq_data.end(), inv_freq_data.begin(), inv_freq_data.end());
            if (inference_dtype_ == base::DType::BF16)
            {
                std::vector<uint16_t> inv_freq_bf16(inv_freq_data.size());
                base::load_f32_to_bf16(inv_freq_data.data(), inv_freq_bf16.data(), inv_freq_bf16.size());
                inv_freq = base::Tensor::from_vector(inv_freq_bf16, {1, head_dim}, device, false, stream_);
            }
            else
            {
                inv_freq = base::Tensor::from_vector(inv_freq_data, {1, head_dim}, device, false, stream_);
            }
        }

        void Qwen3RotaryEmbedding::forward(size_t pos_start, size_t pos_end, base::PosEmb pos_emb)
        {
            kernel::get_gen_rope_kernel(device_)(&inv_freq, pos_start, pos_end, pos_emb.first, pos_emb.second, stream_);
        }
    }
}
