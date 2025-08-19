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
            : Layer(1, 1, device, stream),
              layer_index_(layer_index),
              input_layernorm(config["hidden_size"], config["rms_norm_eps"]),
              post_attention_layernorm(config["hidden_size"], config["rms_norm_eps"]),
              self_attn(layer_index, config, device, stream),
              mlp(layer_index, config, device, stream)
        {
        }

        void Qwen3DecodeLayer::forward()
        {
            // Forward pass logic for Qwen3DecodeLayer
            // This should be implemented based on the specific requirements of the layer
            VLOG(TRACE) << "Forward pass for Qwen3DecodeLayer at index: " << layer_index_;
            throw std::runtime_error("Forward pass not implemented for Qwen3DecodeLayer");
            // Example: Use inputs and outputs tensors to perform computations
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