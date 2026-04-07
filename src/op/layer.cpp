#include "op/layer.h"
#include "base/util.h"
#include "base/allocator.h"

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

        void WLayer::loadWeight(const std::string &name, base::SafeTensors &st, bool transpose)
        {
            name_ = name;
            if (transpose)
            {
                std::vector<size_t> weight_shape = st.get_weight_shape(name_ + ".weight");
                auto this_shape = weight_.shape();
                CHECK(this_shape.size() >= 2);
                std::swap(this_shape[this_shape.size() - 2], this_shape[this_shape.size() - 1]);
                CHECK(this_shape == weight_shape) << "Weight shape mismatch for " << name;
                VLOG(DEBUG) << "transpose " << name << " to load the inverse weight";
                weight_.view(this_shape);
            }

            size_t weight_size = weight_.size();
            if (device_ == base::Device::CPU)
            {
                VLOG(TRACE) << "Loading " << name << " weights to CPU";
                st.materialize_weight(name_ + ".weight", weight_.raw_data(), weight_.dtype());
            }
            else if (device_ == base::Device::CUDA)
            {
                VLOG(TRACE) << "Loading " << name << " weights to CUDA";
                base::ArrBuffer buffer(base::HostAllocator::getInstance(), weight_size * weight_.element_size());
                st.materialize_weight(name_ + ".weight", buffer.data(), weight_.dtype());
                base::Allocator::device_memcpy(weight_.raw_data(), buffer.data(), weight_size * weight_.element_size(), cudaMemcpyHostToDevice);
            }
            if (transpose)
            {
                weight_.t();
                weight_.contiguous();
            }
        }
    }
}
