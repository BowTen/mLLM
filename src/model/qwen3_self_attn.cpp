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
            target_shape[target_shape.size() - 3] = num_attention_heads;
            k.insert_dim(-2);
            v.insert_dim(-2);
            k.expand(-3, num_attention_heads / num_key_value_heads);
            v.expand(-3, num_attention_heads / num_key_value_heads);
            k.reshape(target_shape);
            v.reshape(target_shape);
        }

        Qwen3SelfAttn::Qwen3SelfAttn(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream)
            : layer_index_(layer_index),
              device_(device),
              stream_(stream),
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
              k_cache(),
              v_cache(),
              mat_mul(device_, stream_),
              mat_sc_mul(device_, stream_),
              mat_mul_attn_output(device_, stream_),
              causal_mask(device_, stream_),
              softmax(device_, stream_),
              scaling(Tensor::from_float(static_cast<float>(std::pow(head_dim, -0.5f)), device))
        {
            VLOG(TRACE) << "Constructor: Qwen3SelfAttn with layer index: " << layer_index_;
        }

        void Qwen3SelfAttn::forward(Tensor *hidden_state, Tensor *output, base::PosEmb position_embeddings)
        {
            VLOG(DEBUG) << "Forward pass for Qwen3SelfAttn at index: " << layer_index_;
            std::vector<size_t> q_shape(hidden_state->shape());
            std::vector<size_t> kv_shape(hidden_state->shape());
            q_shape.back() = num_attention_heads * head_dim;
            kv_shape.back() = num_key_value_heads * head_dim;
            if (q_output.shape() != q_shape)
            {
                q_output = Tensor(q_shape, device_, false);
                k_output = Tensor(kv_shape, device_, true); // reapeat_kv需要扩展Tensor
                v_output = Tensor(kv_shape, device_, true); // reapeat_kv需要扩展Tensor
            }

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
            q_norm.forward(q_output, q_output);
            k_norm.forward(k_output, k_output);
            q_output.transpose(-3, -2);
            k_output.transpose(-3, -2);
            v_output.transpose(-3, -2);

            kernel::get_rope_kernel(device_)(&q_output, position_embeddings.first, position_embeddings.second, &q_output, stream_);
            kernel::get_rope_kernel(device_)(&k_output, position_embeddings.first, position_embeddings.second, &k_output, stream_);

            repeat_kv(k_output, v_output);
            CHECK(k_output.shape() == v_output.shape());
            CHECK_EQ(k_output.num_mats(), v_output.num_mats());

            if (k_cache.empty())
            {
                VLOG(DEBUG) << "Creating new KV cache tensors in layer " << layer_index_;
                k_cache = k_output;
                v_cache = v_output;
            }
            else
            {
                k_cache.cat(k_output, -2);
                v_cache.cat(v_output, -2);
            }

            Tensor attn_weights({num_attention_heads, k_cache.shape(-2), k_cache.shape(-2)}, device_);

            k_cache.t();
            mat_mul.forward(q_output, k_cache, attn_weights);
            k_cache.t();
            mat_sc_mul.forward(attn_weights, scaling, attn_weights);
            causal_mask.forward(attn_weights);
            softmax.forward(attn_weights, attn_weights);

            Tensor attn_output(q_output);
            mat_mul_attn_output.forward(attn_weights, v_cache, attn_output);

            attn_output.transpose(-3, -2);
            auto attn_out_shape = attn_output.shape();
            attn_out_shape.pop_back();
            attn_out_shape.back() *= head_dim;
            attn_output.reshape(attn_out_shape);

            o_proj.forward(attn_output, *output);
            CHECK_CUDA_ERR(cudaGetLastError());
        }

        void Qwen3SelfAttn::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            name_ = name;
            q_proj.loadWeight(name_ + ".q_proj", st, true);
            k_proj.loadWeight(name_ + ".k_proj", st, true);
            v_proj.loadWeight(name_ + ".v_proj", st, true);
            o_proj.loadWeight(name_ + ".o_proj", st, true);
            q_norm.loadWeight(name_ + ".q_norm", st, false);
            k_norm.loadWeight(name_ + ".k_norm", st, false);
        }
        std::vector<WLayer *> Qwen3SelfAttn::weighted_layers()
        {
            std::vector<WLayer *> wlayers;
            wlayers.push_back(&q_proj);
            wlayers.push_back(&k_proj);
            wlayers.push_back(&v_proj);
            wlayers.push_back(&o_proj);
            wlayers.push_back(&q_norm);
            wlayers.push_back(&k_norm);
            return wlayers;
        }

    } // namespace model
} // namespace mllm