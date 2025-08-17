#ifndef MLLM_MODEL_QWEN3_H
#define MLLM_MODEL_QWEN3_H

#include "tokenizer/tokenizer.h"
#include "op/embedding.h"

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3
        {
            JsonConfig config;
            size_t vocab_size;
            size_t hidden_size;
            BPETokenizer tokenizer;
            Embedding embed_tokens;

            Qwen3(std::string model_path, base::Device device = base::Device::CPU);

        public:
            static Qwen3 from_pretrained(const std::string &model_path);
        };
    }
}

#endif // MLLM_MODEL_QWEN3_H