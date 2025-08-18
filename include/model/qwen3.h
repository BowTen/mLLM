#ifndef MLLM_MODEL_QWEN3_H
#define MLLM_MODEL_QWEN3_H

#include "tokenizer/tokenizer.h"
#include "op/embedding.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3
        {
            JsonConfig config_;
            size_t vocab_size;
            size_t hidden_size;
            BPETokenizer tokenizer;
            Embedding embed_tokens;
            base::Device device_;
            cudaStream_t stream_;

            Qwen3(std::string model_path, base::Device device = base::Device::CPU);

        public:
            static Qwen3 from_pretrained(const std::string &model_path, base::Device device = base::Device::CPU)
            {
                return Qwen3(model_path, device);
            }

            Tensor forward_test(std::string text);
            JsonConfig config() const { return config_; }
        };
    }
}

#endif // MLLM_MODEL_QWEN3_H