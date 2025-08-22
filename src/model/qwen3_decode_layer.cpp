#include "model/qwen3_decode_layer.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace model
    {
        Qwen3DecodeLayer::Qwen3DecodeLayer(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream)
            : layer_index_(layer_index),
              input_layernorm(config["hidden_size"], config["rms_norm_eps"]),
              post_attention_layernorm(config["hidden_size"], config["rms_norm_eps"]),
              self_attn(layer_index, config, device, stream),
              mlp(layer_index, config, device, stream),
              add_op(device, stream)
        {
        }

        void Qwen3DecodeLayer::forward(Tensor *hidden_state, Tensor *output, base::PosEmb position_embeddings)
        {
            VLOG(TRACE) << "Forward pass for Qwen3DecodeLayer at index: " << layer_index_;
            auto residual = hidden_state->clone();

            input_layernorm.forward(*hidden_state, *hidden_state);
            self_attn.forward(hidden_state, hidden_state, position_embeddings);
            add_op.forward(*hidden_state, residual, *hidden_state);

            residual = hidden_state->clone();

            post_attention_layernorm.forward(*hidden_state, *hidden_state);
            mlp.forward(*hidden_state, *hidden_state);
            add_op.forward(*hidden_state, residual, *output);
        }

        void Qwen3DecodeLayer::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            VLOG(TRACE) << "Loading weights for Qwen3DecodeLayer: " << name;
            name_ = name;
            input_layernorm.loadWeight(name_ + ".input_layernorm", st);
            post_attention_layernorm.loadWeight(name_ + ".post_attention_layernorm", st);
            self_attn.loadWeight(name_ + ".self_attn", st);
            mlp.loadWeight(name_ + ".mlp", st);
            VLOG(TRACE) << "Successfully loaded weights for Qwen3DecodeLayer: " << name_;
        }
    } // namespace model
} // namespace mllm