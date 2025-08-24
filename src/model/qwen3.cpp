#include "model/qwen3.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"
#include <utility>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace model
    {
        cudaStream_t Qwen3::init_cuda_stream(base::Device device)
        {
            cudaStream_t stream = nullptr;
            if (device == base::Device::CUDA)
            {
                VLOG(TRACE) << "Initializing CUDA stream";
                cudaStreamCreate(&stream);
            }
            return stream;
        }

        Qwen3::Qwen3(std::string model_path, base::Device device, float temperature)
            : Layer(1, 1, device, init_cuda_stream(device)),
              config_(base::load_json(model_path + "/config.json")),
              vocab_size(config_["vocab_size"]),
              hidden_size(config_["hidden_size"]),
              tokenizer(BPETokenizer::from_file(model_path + "/tokenizer.json")),
              embed_tokens(vocab_size, hidden_size, device_, stream_),
              rotary_embedding(config_, device_, stream_),
              norm(hidden_size, config_["rms_norm_eps"], device_, stream_),
              temp_scal(device_, stream_),
              lm_head({hidden_size, vocab_size}, device_, stream_),
              softmax(device_, stream_),
              pos_id(0),
              temperature_scaling(base::Tensor::from_float(1.0f / temperature, device_, stream_)),
              final_probability({1, vocab_size}, device_, stream_)
        {
            VLOG(TRACE) << "Loading Qwen3 model from: " << model_path;
            VLOG(TRACE) << "Loading safetensors from: " << model_path + "/model.safetensors";
            // 加载safetensors
            base::SafeTensors st(model_path + "/model.safetensors");
            VLOG(TRACE) << "Success to load safetensors";
            auto header = st.get_header();

            embed_tokens.loadWeight("model.embed_tokens", st);
            norm.loadWeight("model.norm", st);
            lm_head.loadWeight("lm_head", st);

            VLOG(TRACE) << "Loading layers";
            for (size_t i = 0; i < config_["num_hidden_layers"]; ++i)
            {
                layers.emplace_back(i, config_, device, stream_);
                layers.back().loadWeight("model.layers." + std::to_string(i), st);
            }
            VLOG(TRACE) << "Successfully loaded all weights";
        }

        void Qwen3::forward(Tensor &token_ids, Tensor &next_token_id)
        {
            setInput(0, token_ids);
            setOutput(0, next_token_id);
            VLOG(TRACE) << "Forward pass through Qwen3 model";

            hidden_state = Tensor({token_ids.shape(0), hidden_size}, device_);
            embed_tokens.forward(token_ids, hidden_state);

            auto rope_emb_shape = hidden_state.shape();
            rope_emb_shape.back() = config_["head_dim"];
            cos = Tensor(rope_emb_shape, device_);
            sin = Tensor(rope_emb_shape, device_);
            base::PosEmb pos_emb(&cos, &sin);
            size_t seq_len = hidden_state.shape(-2);
            rotary_embedding.forward(pos_id, pos_id + seq_len, pos_emb);

            // Forward pass through each decode layer
            for (auto &layer : layers)
            {
                layer.forward(&hidden_state, &hidden_state, pos_emb);
            }

            kernel::get_last_hidden_state_kernel(device_)(&hidden_state, nullptr);

            norm.forward(hidden_state, hidden_state);
            temp_scal.forward(hidden_state, temperature_scaling, hidden_state);

            lm_head.forward(hidden_state, final_probability);
            softmax.forward(final_probability, final_probability);

            kernel::random_sampling_cpu(&final_probability, &getOutput(0), device_);
        }
    } // namespace model
} // namespace mllm