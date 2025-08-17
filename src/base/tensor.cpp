#include <iostream>
#include <vector>
#include "base/tensor.h"
#include "base/allocator.h"
#include "base/buffer.h"

namespace mllm
{
    namespace base
    {

        Tensor::Tensor(const std::vector<size_t> &shape, Device device, bool mut)
            : shape_(shape), device_(device)
        {
            size_t expected_size = 1;
            for (auto dim : shape)
            {
                expected_size *= dim;
            }

            Allocator *allocator = nullptr;
            if (device == Device::CPU)
                allocator = HostAllocator::getInstance();
            else if (device == Device::CUDA)
                allocator = CudaAllocator::getInstance();
            else
                throw std::invalid_argument("Unsupported device type.");
            if (mut)
                buffer_ = std::make_shared<VecBuffer>(allocator, expected_size * sizeof(float) * 2, expected_size * sizeof(float));
            else
                buffer_ = std::make_shared<ArrBuffer>(allocator, expected_size * sizeof(float));
        }

        const std::vector<size_t> &Tensor::shape() const
        {
            return shape_;
        }

        size_t Tensor::size() const
        {
            if (!buffer_)
                return 0;
            return buffer_->size() / sizeof(float);
        }

        float *Tensor::data()
        {
            if (!buffer_)
                return nullptr;
            return static_cast<float *>(buffer_->data());
        }
    } // namespace base
} // namespace mllm