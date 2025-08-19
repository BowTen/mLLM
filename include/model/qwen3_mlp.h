#ifndef MLLM_MODEL_QWEN3_MLP_H
#define MLLM_MODEL_QWEN3_MLP_H

#include "tokenizer/tokenizer.h"
#include "base/safetensors.h"
#include "op/rms_norm.h"
#include "op/linear.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace model
    {

        using namespace tokenizer;
        using namespace op;

        class Qwen3MLP : public Layer
        {
            size_t layer_index_;
            JsonConfig config_;
            size_t hidden_size;
            size_t intermediate_size;
            Linear gate_proj;
            Linear up_proj;
            Linear down_proj;

        public:
            Qwen3MLP(size_t layer_index, JsonConfig config, base::Device device = base::Device::CPU, cudaStream_t stream = nullptr);
            void forward() override;
            using Layer::forward;
            void loadWeight(const std::string &name, base::SafeTensors &st);
        };
    }
}

#endif // MLLM_MODEL_QWEN3_MLP_H