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

            // 加载权重
            load_weight_for_embed_tokens(st);
            load_weight_for_norm(st);
            VLOG(TRACE) << "Successfully loaded all weights";
        }

        void Qwen3::load_weight_for_embed_tokens(base::SafeTensors &st)
        {
            if (device_ == base::Device::CUDA)
            {
                embed_tokens.setStream(stream_);
            }
            auto embed_weigth = embed_tokens.getWeight().data();
            if (device_ == base::Device::CPU)
            {
                VLOG(TRACE) << "Loading embed_tokens weights to CPU";
                // CPU设备，直接加载权重
                base::load_bf16_to_f32(st.get_weight("model.embed_tokens.weight"), embed_weigth, vocab_size * hidden_size);
            }
            else if (device_ == base::Device::CUDA)
            {
                VLOG(TRACE) << "Loading embed_tokens weights to CUDA";
                // CUDA设备，先拷贝到临时缓冲区处理
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), vocab_size * hidden_size * sizeof(float));
                base::load_bf16_to_f32(st.get_weight("model.embed_tokens.weight"), buffer.data(), vocab_size * hidden_size);
                cudaMemcpy(embed_weigth, buffer.data(), vocab_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            VLOG(TRACE) << "Successfully loaded embed_tokens weights";
        }

        void Qwen3::load_weight_for_norm(base::SafeTensors &st)
        {
            if (device_ == base::Device::CUDA)
            {
                norm.setStream(stream_);
            }
            auto norm_weight = norm.getWeight().data();
            if (device_ == base::Device::CPU)
            {
                VLOG(TRACE) << "Loading norm weights to CPU";
                // CPU设备，直接加载权重
                base::load_bf16_to_f32(st.get_weight("model.norm.weight"), norm_weight, hidden_size);
            }
            else if (device_ == base::Device::CUDA)
            {
                VLOG(TRACE) << "Loading norm weights to CUDA";
                // CUDA设备，先拷贝到临时缓冲区处理
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), hidden_size * sizeof(float));
                base::load_bf16_to_f32(st.get_weight("model.norm.weight"), buffer.data(), hidden_size);
                cudaMemcpy(norm_weight, buffer.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            VLOG(TRACE) << "Successfully loaded norm weights";
        }
    } // namespace model
} // namespace mllm