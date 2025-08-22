#ifndef MLLM_MODEL_QWEN3_H
#define MLLM_MODEL_QWEN3_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "op/embedding.h"
#include "op/rms_norm.h"
#include "qwen3_decode_layer.h"
#include "qwen3_rotary_embedding.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3 : public op::Layer
        {
            JsonConfig config_;
            size_t vocab_size;
            size_t hidden_size;
            BPETokenizer tokenizer;
            Embedding embed_tokens;
            RMSNorm norm;
            Qwen3RotaryEmbedding rotary_embedding;
            std::vector<Qwen3DecodeLayer> layers;
            size_t pos_id;

            Qwen3(std::string model_path, base::Device device = base::Device::CPU);

        public:
            static Qwen3 from_pretrained(const std::string &model_path, base::Device device = base::Device::CPU)
            {
                return Qwen3(model_path, device);
            }

            void forward() override;

            JsonConfig config() const { return config_; }
            BPETokenizer *get_tokenizer() { return &tokenizer; }
            Embedding *get_embed_tokens() { return &embed_tokens; }
            RMSNorm *get_norm() { return &norm; }
            base::Device device() const { return device_; }
            cudaStream_t stream() const { return stream_; }
        };
    }
}

#endif // MLLM_MODEL_QWEN3_H