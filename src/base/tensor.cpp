#include <iostream>
#include <vector>
#include "base/tensor.h"
#include "base/allocator.h"
#include "base/buffer.h"
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {

        bool isDevicePointer(void *ptr)
        {
            cudaPointerAttributes attributes;
            cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

            if (err == cudaSuccess)
            {
                return (attributes.type == cudaMemoryTypeDevice);
            }
            return false;
        }

        Tensor::Tensor(const std::vector<size_t> &shape, Device device, bool mut)
            : shape_(shape), device_(device), mut_(mut)
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

        Tensor::Tensor(void *data, const std::vector<size_t> &shape, Device device, bool mut)
            : shape_(shape), device_(device), mut_(mut)
        {
            if (isDevicePointer(data) != (device == Device::CUDA))
            {
                throw std::invalid_argument("Data pointer device type does not match Tensor device type.");
            }
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
                buffer_ = std::make_shared<VecBuffer>(allocator, data, expected_size * sizeof(float), expected_size * sizeof(float));
            else
                buffer_ = std::make_shared<ArrBuffer>(allocator, data, expected_size * sizeof(float));
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

        void Tensor::toDevice(Device device)
        {
            if (device_ == device)
                return;
            if (!buffer_)
            {
                LOG(WARNING) << "Tensor buffer is empty, cannot transfer device.";
                return;
            }
            VLOG(DEBUG) << "Transferring tensor from device " << (device_ == Device::CPU ? "CPU" : "CUDA") << " to " << (device == Device::CPU ? "CPU" : "CUDA");

            Allocator *new_allocator = nullptr;
            if (device == Device::CPU)
                new_allocator = HostAllocator::getInstance();
            else if (device == Device::CUDA)
                new_allocator = CudaAllocator::getInstance();
            else
                throw std::invalid_argument("Unsupported device type.");
            Buffer::BufferPtr new_buffer;
            if (mut_)
                new_buffer = std::make_shared<VecBuffer>(new_allocator, buffer_->size() * 2, buffer_->size());
            else
                new_buffer = std::make_shared<ArrBuffer>(new_allocator, buffer_->size());
            cudaMemcpy(new_buffer->data(), buffer_->data(), buffer_->size(), device == Device::CPU ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
            buffer_ = new_buffer;
            device_ = device;
            VLOG(DEBUG) << "Tensor transferred successfully.";
        }
    } // namespace base
} // namespace mllm