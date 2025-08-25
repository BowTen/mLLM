#ifndef MLLM_MODEL_QWEN3_H
#define MLLM_MODEL_QWEN3_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "op/embedding.h"
#include "op/rms_norm.h"
#include "op/mat_mul.h"
#include "op/linear.h"
#include "op/softmax.h"
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
        private:
            JsonConfig config_;
            size_t vocab_size;
            size_t hidden_size;
            BPETokenizer tokenizer;
            Embedding embed_tokens;
            Qwen3RotaryEmbedding rotary_embedding;
            std::vector<Qwen3DecodeLayer> layers;
            RMSNorm norm;
            MatMul temp_scal;
            Linear lm_head;
            Softmax softmax;

            size_t pos_id;
            Tensor hidden_state;
            Tensor cos;
            Tensor sin;
            Tensor temperature_scaling;
            Tensor final_probability;

            Qwen3(std::string model_path, base::Device device, float temperature);

            cudaStream_t init_cuda_stream(base::Device device);
            void print_top_tokens_cpu(Tensor &probabilities, size_t top_k);

        public:
            static Qwen3 from_pretrained(const std::string &model_path, base::Device device, float temperature = 1.0f)
            {
                return Qwen3(model_path, device, temperature);
            }

            void forward(Tensor &token_ids, Tensor &next_token_id);

            JsonConfig config() const { return config_; }
            BPETokenizer *get_tokenizer() { return &tokenizer; }
            Embedding *get_embed_tokens() { return &embed_tokens; }
            Qwen3RotaryEmbedding *get_rotary_embedding() { return &rotary_embedding; }
            std::vector<Qwen3DecodeLayer> *get_layers() { return &layers; }
            RMSNorm *get_norm() { return &norm; }
            MatMul *get_temp_scal() { return &temp_scal; }
            Linear *get_lm_head() { return &lm_head; }
            base::Device device() const { return device_; }
            cudaStream_t stream() const { return stream_; }

            std::vector<WLayer *> weighted_layers();
            void register_hooks(WLayer::Hook hook);
        };
    }
}

#endif // MLLM_MODEL_QWEN3_H