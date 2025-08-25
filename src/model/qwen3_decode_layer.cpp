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
              input_layernorm(config["hidden_size"], config["rms_norm_eps"], device, stream),
              post_attention_layernorm(config["hidden_size"], config["rms_norm_eps"], device, stream),
              self_attn(layer_index, config, device, stream),
              mlp(layer_index, config, device, stream),
              add_op(device, stream)
        {
        }

        void Qwen3DecodeLayer::forward(Tensor *hidden_state, Tensor *output, base::PosEmb position_embeddings)
        {
            VLOG(TRACE) << "Forward pass for Qwen3DecodeLayer at index: " << layer_index_;

            // TODO: 可优化，如果已有空间则直接拷贝数据
            attn_residual = hidden_state->clone();

            input_layernorm.forward(*hidden_state, *hidden_state);
            self_attn.forward(hidden_state, hidden_state, position_embeddings);
            add_op.forward(*hidden_state, attn_residual, *hidden_state);

            // TODO: 可优化，如果已有空间则直接拷贝数据
            mlp_residual = hidden_state->clone();

            post_attention_layernorm.forward(*hidden_state, *hidden_state);
            mlp.forward(hidden_state, hidden_state);
            add_op.forward(*hidden_state, mlp_residual, *output);
        }

        void Qwen3DecodeLayer::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            name_ = name;
            input_layernorm.loadWeight(name_ + ".input_layernorm", st, false);
            post_attention_layernorm.loadWeight(name_ + ".post_attention_layernorm", st, false);
            self_attn.loadWeight(name_ + ".self_attn", st);
            mlp.loadWeight(name_ + ".mlp", st);
        }

        std::vector<WLayer *> Qwen3DecodeLayer::weighted_layers()
        {
            std::vector<WLayer *> wlayers;
            wlayers.push_back(&input_layernorm);
            wlayers.push_back(&post_attention_layernorm);
            auto attn_wlayer = self_attn.weighted_layers();
            wlayers.insert(wlayers.end(), attn_wlayer.begin(), attn_wlayer.end());
            auto mlp_wlayer = mlp.weighted_layers();
            wlayers.insert(wlayers.end(), mlp_wlayer.begin(), mlp_wlayer.end());
            return wlayers;
        }

    } // namespace model
} // namespace mllm