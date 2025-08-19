#include "model/qwen3.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace model
    {

        Qwen3::Qwen3(std::string model_path, base::Device device)
            : config_(base::load_json(model_path + "/config.json")),
              vocab_size(config_["vocab_size"]),
              hidden_size(config_["hidden_size"]),
              tokenizer(BPETokenizer::from_file(model_path + "/tokenizer.json")),
              embed_tokens(vocab_size, hidden_size, device),
              norm(hidden_size, config_["rms_norm_eps"], device),
              device_(device),
              stream_(nullptr)
        {
            VLOG(TRACE) << "Loading Qwen3 model from: " << model_path;
            // 初始化CUDA流
            if (device == base::Device::CUDA)
            {
                VLOG(TRACE) << "Initializing CUDA stream";
                cudaStreamCreate(&stream_);
            }
            VLOG(TRACE) << "Loading safetensors from: " << model_path + "/model.safetensors";
            // 加载safetensors
            base::SafeTensors st(model_path + "/model.safetensors");
            VLOG(TRACE) << "Success to load safetensors";
            auto header = st.get_header();

            embed_tokens.setStream(stream_);
            norm.setStream(stream_);
            embed_tokens.loadWeight("model.embed_tokens", st);
            norm.loadWeight("model.norm", st);

            VLOG(TRACE) << "Loading layers";
            for (size_t i = 0; i < config_["num_hidden_layers"]; ++i)
            {
                layers.emplace_back(i, config_, device, stream_);
                layers.back().loadWeight("model.layers." + std::to_string(i), st);
            }
            VLOG(TRACE) << "Successfully loaded all weights";
        }
    } // namespace model
} // namespace mllm