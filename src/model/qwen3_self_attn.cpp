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
        Qwen3SelfAttn::Qwen3SelfAttn(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream)
            : Layer(1, 1, device, stream),
              layer_index_(layer_index),
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
              k_norm(head_dim, config["rms_norm_eps"], device, stream)
        {
        }

        void Qwen3SelfAttn::forward()
        {
            VLOG(TRACE) << "Forward pass for Qwen3SelfAttn at index: " << layer_index_;
            auto hidden_state = this->getInput(0);
            Tensor q_output({hidden_state.shape()[0], num_attention_heads * head_dim}, device_, stream_);
            Tensor k_output({hidden_state.shape()[0], num_key_value_heads * head_dim}, device_, stream_);
            Tensor v_output({hidden_state.shape()[0], num_key_value_heads * head_dim}, device_, stream_);

            q_proj.forward(hidden_state, q_output);
            k_proj.forward(hidden_state, k_output);
            v_proj.forward(hidden_state, v_output);

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