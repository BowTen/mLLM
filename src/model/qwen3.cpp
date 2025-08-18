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
              device_(device)
        {
            VLOG(1) << "Loading Qwen3 model from: " << model_path;
            VLOG(1) << "Loading safetensors from: " << model_path + "/model.safetensors";
            // 加载safetensors
            base::SafeTensors st(model_path + "/model.safetensors");
            VLOG(1) << "Success to load safetensors";
            auto header = st.get_header();

            auto embed_weigth = embed_tokens.getWeight().data();
            if (device == base::Device::CPU)
            {
                VLOG(1) << "Loading embed_tokens weights to CPU";
                // CPU设备，直接加载权重
                base::load_bf16_to_f32(st.get_weight("model.embed_tokens.weight"), embed_weigth, vocab_size * hidden_size);
            }
            else if (device == base::Device::CUDA)
            {
                VLOG(1) << "Loading embed_tokens weights to CUDA";
                // CUDA设备，先拷贝到临时缓冲区处理
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), vocab_size * hidden_size * sizeof(float));
                base::load_bf16_to_f32(st.get_weight("embed_tokens"), buffer.data(), vocab_size * hidden_size);
                cudaMemcpy(embed_weigth, buffer.data(), vocab_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            VLOG(1) << "Successfully loaded embed_tokens weights";
        }

        Qwen3 Qwen3::from_pretrained(const std::string &model_path)
        {
            return Qwen3(model_path);
        }

        Tensor Qwen3::forward_test(std::string text)
        {
            auto tokens = tokenizer.encode(text);
            std::vector<size_t> shape = {tokens.size(), 1};
            Tensor input(shape, device_);
            uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                input_data[i] = tokens[i];
            }

            std::vector<size_t> out_shape = {tokens.size(), hidden_size};
            Tensor output(out_shape, device_);

            embed_tokens.setInput(0, input);
            embed_tokens.setOutput(0, output);
            embed_tokens.forward();

            return output;
        }

    } // namespace model
} // namespace mllm