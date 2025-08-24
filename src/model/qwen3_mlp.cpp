#include "model/qwen3_mlp.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace model
    {
        Qwen3MLP::Qwen3MLP(size_t layer_index, JsonConfig config, base::Device device, cudaStream_t stream)
            : layer_index_(layer_index),
              config_(config),
              hidden_size(config["hidden_size"]),
              intermediate_size(config["intermediate_size"]),
              gate_proj({hidden_size, intermediate_size}, device, stream),
              up_proj({hidden_size, intermediate_size}, device, stream),
              down_proj({intermediate_size, hidden_size}, device, stream)
        {
        }

        void Qwen3MLP::forward(Tensor *hidden_state, Tensor *output)
        {
            VLOG(TRACE) << "Forward pass for Qwen3MLP at index: " << layer_index_;
        }

        void Qwen3MLP::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            VLOG(TRACE) << "Loading weights for Qwen3MLP: " << name;
            name_ = name;
            gate_proj.loadWeight(name_ + ".gate_proj", st);
            up_proj.loadWeight(name_ + ".up_proj", st);
            down_proj.loadWeight(name_ + ".down_proj", st);
            VLOG(TRACE) << "Successfully loaded weights for Qwen3MLP: " << name_;
        }
    } // namespace model
} // namespace mllm