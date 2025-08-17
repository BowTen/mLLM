#include "op/layer.h"

namespace mllm
{
    namespace op
    {
        using base::Tensor;

        void Layer::setInput(size_t index, const Tensor &tensor)
        {
            if (index >= inputs.size())
            {
                throw std::out_of_range("Input index out of range");
            }
            inputs[index] = tensor;
        }

        void Layer::setOutput(size_t index, const Tensor &tensor)
        {
            if (index >= outputs.size())
            {
                throw std::out_of_range("Output index out of range");
            }
            outputs[index] = tensor;
        }

        Tensor &Layer::getInput(size_t index)
        {
            if (index >= inputs.size())
            {
                throw std::out_of_range("Input index out of range");
            }
            return inputs.at(index);
        }
        Tensor &Layer::getOutput(size_t index)
        {
            if (index >= outputs.size())
            {
                throw std::out_of_range("Output index out of range");
            }
            return outputs.at(index);
        }
    }
}