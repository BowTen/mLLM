#include "model/qwen3.h"
#include "base/util.h"
#include "base/safetensors.h"
#include "cuda_runtime.h"

namespace mllm
{
    namespace model
    {

        Qwen3::Qwen3(std::string model_path, base::Device device)
            : config(base::load_json(model_path + "/config.json")),
              vocab_size(config["vocab_size"]),
              hidden_size(config["hidden_size"]),
              tokenizer(BPETokenizer::from_file(model_path)),
              embed_tokens(vocab_size, hidden_size, device)
        {
            // 加载safetensors
            base::SafeTensors st(model_path + "/model.safetensors");
            auto header = st.get_header();

            auto embed_weigth = embed_tokens.getWeight().data();
            if (device == base::Device::CPU)
            {
                // CPU设备，直接加载权重
                base::load_bf16_to_f32(st.get_weight("embed_tokens"), embed_weigth, vocab_size * hidden_size);
            }
            else if (device == base::Device::CUDA)
            {
                // CUDA设备，先拷贝到临时缓冲区处理
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), vocab_size * hidden_size * sizeof(float));
                base::load_bf16_to_f32(st.get_weight("embed_tokens"), buffer.data(), vocab_size * hidden_size);
                cudaMemcpy(embed_weigth, buffer.data(), vocab_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
            }
        }

        Qwen3 Qwen3::from_pretrained(const std::string &model_path)
        {
            return Qwen3(model_path);
        }

    } // namespace model
} // namespace mllm