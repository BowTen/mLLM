#include "model/qwen3_self_attn.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"
#include "kernel/kernel.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace model
    {
        void Qwen3SelfAttn::repeat_kv(Tensor &k, Tensor &v)
        {
            auto target_shape = k.shape();
            target_shape[target_shape.size() - 2] = num_attention_heads;
            k.insert_dim(-2);
            v.insert_dim(-2);
            k.expand(-2, num_attention_heads / num_key_value_heads);
            v.expand(-2, num_attention_heads / num_key_value_heads);
            k.reshape(target_shape);
            v.reshape(target_shape);
        }

        Qwen3SelfAttn::Qwen3SelfAttn(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream)
            : layer_index_(layer_index),
              config_(config),
              hidden_size(config["hidden_size"]),
              head_dim(config["head_dim"]),
              num_attention_heads(config["num_attention_heads"]),
              num_key_value_heads(config["num_key_value_heads"]),
              q_proj({hidden_size, num_attention_heads * head_dim}, device, stream),
              k_proj({hidden_size, num_key_value_heads * head_dim}, device, stream),
              v_proj({hidden_size, num_key_value_heads * head_dim}, device, stream),
              o_proj({num_attention_heads * head_dim, hidden_size}, device, stream),
              q_norm(head_dim, config["rms_norm_eps"], device, stream),
              k_norm(head_dim, config["rms_norm_eps"], device, stream),
              k_cache({num_attention_heads, 0, head_dim}, device, true),
              v_cache({num_attention_heads, 0, head_dim}, device, true),
              mat_mul(device_, stream_),
              mat_sc_mul(device_, stream_),
              scaling(std::vector<float>({static_cast<float>(std::pow(head_dim, -0.5f))}).data(), {1}, true, base::Device::CPU)
        {
        }

        void Qwen3SelfAttn::forward(Tensor *hidden_state, Tensor *output, base::PosEmb position_embeddings)
        {
            VLOG(TRACE) << "Forward pass for Qwen3SelfAttn at index: " << layer_index_;
            std::vector<size_t> q_shape(hidden_state->shape());
            std::vector<size_t> kv_shape(hidden_state->shape());
            q_shape.back() = num_attention_heads * head_dim;
            kv_shape.back() = num_key_value_heads * head_dim;
            Tensor q_output(q_shape, device_, false);
            Tensor k_output(kv_shape, device_, true);
            Tensor v_output(kv_shape, device_, true);

            q_proj.forward(*hidden_state, q_output);
            k_proj.forward(*hidden_state, k_output);
            v_proj.forward(*hidden_state, v_output);

            q_shape.back() = num_attention_heads;
            q_shape.push_back(head_dim);
            kv_shape.back() = num_key_value_heads;
            kv_shape.push_back(head_dim);

            q_output.view(q_shape);
            k_output.view(kv_shape);
            v_output.view(kv_shape);
            q_output.transpose(-3, -2);
            k_output.transpose(-3, -2);
            v_output.transpose(-3, -2);

            q_norm.forward(q_output, q_output);
            k_norm.forward(k_output, k_output);

            kernel::get_rope_kernel(device_)(&q_output, position_embeddings.first, position_embeddings.second, &q_output, stream_);
            kernel::get_rope_kernel(device_)(&k_output, position_embeddings.first, position_embeddings.second, &k_output, stream_);

            repeat_kv(k_output, v_output);

            k_cache.cat(k_output, -2);
            v_cache.cat(v_output, -2);

            Tensor attn_weights({k_cache.shape(-2), k_cache.shape(-2)}, device_);
            k_cache.t();

            mat_mul.forward(q_output, k_cache, attn_weights);
            mat_sc_mul.forward(attn_weights, scaling, attn_weights);

            throw std::runtime_error("Forward pass not implemented for Qwen3SelfAttn");
        }

        void Qwen3SelfAttn::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            VLOG(TRACE) << "Loading weights for Qwen3SelfAttn: " << name;
            name_ = name;
            q_proj.loadWeight(name_ + ".q_proj", st);
            k_proj.loadWeight(name_ + ".k_proj", st);
            v_proj.loadWeight(name_ + ".v_proj", st);
            o_proj.loadWeight(name_ + ".o_proj", st);
            q_norm.loadWeight(name_ + ".q_norm", st);
            k_norm.loadWeight(name_ + ".k_norm", st);
            VLOG(TRACE) << "Successfully loaded weights for Qwen3SelfAttn: " << name_;
        }
    } // namespace model
} // namespace mllm