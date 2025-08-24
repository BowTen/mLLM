#include "op/layer.h"
#include "base/util.h"

namespace mllm
{
    namespace op
    {
        using base::Tensor;

        void Layer::forward(base::Tensor &input)
        {
            setInput(0, input);
            forward();
        }
        void Layer::forward(base::Tensor &input, base::Tensor &output)
        {
            setInput(0, input);
            setOutput(0, output);
            forward();
        }
        void Layer::forward(base::Tensor &input0, base::Tensor &input1, base::Tensor &output)
        {
            setInput(0, input0);
            setInput(1, input1);
            setOutput(0, output);
            forward();
        }

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

        void WLayer::loadWeight(const std::string &name, base::SafeTensors &st)
        {
            name_ = name;
            auto weight_data = weight_.data();
            size_t weight_size = weight_.size();
            if (device_ == base::Device::CPU)
            {
                VLOG(TRACE) << "Loading " << name << " weights to CPU";
                // CPU设备，直接加载权重
                base::load_bf16_to_f32(st.get_weight(name_ + ".weight"), weight_data, weight_size);
            }
            else if (device_ == base::Device::CUDA)
            {
                VLOG(TRACE) << "Loading " << name << " weights to CUDA";
                // CUDA设备，先拷贝到临时缓冲区处理
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), weight_size * sizeof(float));
                base::load_bf16_to_f32(st.get_weight(name_ + ".weight"), buffer.data(), weight_size);
                cudaMemcpy(weight_data, buffer.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
            }
            VLOG(TRACE) << "Successfully loaded " << name << " weights";
        }
    }
}