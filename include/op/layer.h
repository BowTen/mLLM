#ifndef MLLM_OP_LAYER_H
#define MLLM_OP_LAYER_H

#include "base/tensor.h"
#include "base/json.hpp"

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

            Layer(size_t input_count, size_t output_count)
                : inputs(input_count), outputs(output_count) {}

        public:
            virtual ~Layer() = default;

            virtual void forward() = 0;

            void setInput(size_t index, const Tensor &tensor);
            void setOutput(size_t index, const Tensor &tensor);
            Tensor &getInput(size_t index);
            Tensor &getOutput(size_t index);
            void setName(const std::string &name) { name_ = name; }
            const std::string &name() const { return name_; }
        };

        class WLayer : public Layer
        {
        protected:
            Tensor weight_;

            WLayer(size_t input_count,
                   size_t output_count,
                   const std::vector<size_t> &shape,
                   base::Device device = base::Device::CPU) : Layer(input_count, output_count),
                                                              weight_(Tensor(shape, device)) {}

        public:
            const Tensor &weight() const { return weight_; }
            void setWeight(const Tensor &weight) { weight_ = weight; }
            Tensor &getWeight() { return weight_; }
        };

    }
}

#endif // MLLM_OP_LAYER_H