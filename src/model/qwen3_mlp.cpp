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
              device_(device),
              config_(config),
              hidden_size(config["hidden_size"]),
              intermediate_size(config["intermediate_size"]),
              gate_proj({hidden_size, intermediate_size}, device, stream),
              up_proj({hidden_size, intermediate_size}, device, stream),
              down_proj({intermediate_size, hidden_size}, device, stream),
              silu(device, stream),
              ele_mul(device, stream)
        {
        }

        void Qwen3MLP::forward(Tensor *hidden_state, Tensor *output)
        {
            VLOG(TRACE) << "Forward pass for Qwen3MLP at index: " << layer_index_;

            auto intermediate_shape = hidden_state->shape();
            intermediate_shape.back() = intermediate_size;
            if (gate_state.shape() != intermediate_shape)
            {
                gate_state = Tensor(intermediate_shape, device_);
                up_state = Tensor(intermediate_shape, device_);
            }

            up_proj.forward(*hidden_state, up_state);
            gate_proj.forward(*hidden_state, gate_state);
            silu.forward(gate_state);

            ele_mul.forward(gate_state, up_state, gate_state);

            down_proj.forward(gate_state, *output);
        }

        void Qwen3MLP::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            name_ = name;
            gate_proj.loadWeight(name_ + ".gate_proj", st, true);
            up_proj.loadWeight(name_ + ".up_proj", st, true);
            down_proj.loadWeight(name_ + ".down_proj", st, true);
        }

        std::vector<WLayer *> Qwen3MLP::weighted_layers()
        {
            std::vector<WLayer *> wlayers;
            wlayers.push_back(&gate_proj);
            wlayers.push_back(&up_proj);
            wlayers.push_back(&down_proj);
            return wlayers;
        }
    } // namespace model
} // namespace mllm