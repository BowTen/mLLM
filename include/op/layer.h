#ifndef MLLM_OP_LAYER_H
#define MLLM_OP_LAYER_H

#include "base/tensor.h"
#include "base/json.hpp"
#include "base/safetensors.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace op
    {
        using JsonConfig = nlohmann::json;
        using base::Tensor;

        class Layer
        {
        protected:
            std::vector<Tensor> inputs;
            std::vector<Tensor> outputs;
            std::string name_;
            base::Device device_;
            cudaStream_t stream_;

            Layer(size_t input_count, size_t output_count, base::Device device, cudaStream_t stream)
                : inputs(input_count), outputs(output_count), device_(device), stream_(stream) {}

        public:
            virtual ~Layer() = default;

            void setInput(size_t index, const Tensor &tensor);
            void setOutput(size_t index, const Tensor &tensor);
            Tensor &getInput(size_t index);
            Tensor &getOutput(size_t index);
            void setName(const std::string &name) { name_ = name; }
            const std::string &name() const { return name_; }
            void setStream(cudaStream_t stream) { stream_ = stream; }
            cudaStream_t stream() const { return stream_; }
        };

        class WLayer : public Layer
        {
        protected:
            Tensor weight_;

            WLayer(size_t input_count,
                   size_t output_count,
                   const std::vector<size_t> &shape,
                   base::Device device,
                   cudaStream_t stream) : Layer(input_count, output_count, device, stream),
                                          weight_(Tensor(shape, device)) {}

        public:
            const Tensor &weight() const { return weight_; }
            void setWeight(const Tensor &weight) { weight_ = weight; }
            Tensor &getWeight() { return weight_; }
            void loadWeight(const std::string &name, base::SafeTensors &st, bool transpose);
        };

    }
}

#endif // MLLM_OP_LAYER_H