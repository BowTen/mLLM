#include <iostream>
#include <vector>
#include "base/tensor.h"
#include "base/allocator.h"
#include "base/buffer.h"
#include "kernel/kernel.h"
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        std::vector<size_t> Tensor::default_stride(const std::vector<size_t> &shape)
        {
            if (shape.empty())
                return {};
            std::vector<size_t> stride(shape.size());
            stride.back() = 1;
            for (int i = shape.size() - 2; i >= 0; --i)
            {
                stride[i] = stride[i + 1] * shape[i + 1];
            }
            return stride;
        }
        Tensor::Tensor() : shape_(),
                           stride_(default_stride(shape_)),
                           is_contiguous_(true),
                           buffer_(nullptr),
                           device_(Device::CPU)
        {
            update();
        }
        Tensor::Tensor(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device, bool mut)
            : shape_(shape),
              is_contiguous_(true),
              buffer_(buffer),
              device_(device),
              mut_(mut)
        {
            if (shape_.size() == 1)
                shape_.insert(shape_.begin(), 1);
            stride_ = default_stride(shape_);
            update();
        }

        Tensor::Tensor(const std::vector<size_t> &shape, Device device, bool mut)
            : shape_(shape), is_contiguous_(true), device_(device), mut_(mut)
        {
            if (shape_.size() == 1)
                shape_.insert(shape_.begin(), 1);
            stride_ = default_stride(shape_);
            size_t expected_size = 1;
            for (auto dim : shape_)
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
            update();
        }

        Tensor::Tensor(void *data, const std::vector<size_t> &shape, bool copy, Device device, bool mut)
            : shape_(shape), is_contiguous_(true), device_(device), mut_(mut)
        {
            if (shape_.size() == 1)
                shape_.insert(shape_.begin(), 1);
            stride_ = default_stride(shape_);
            if (isDevicePointer(data) != (device == Device::CUDA))
            {
                throw std::invalid_argument("Data pointer device type does not match Tensor device type.");
            }
            size_t expected_size = 1;
            for (auto dim : shape_)
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
                buffer_ = std::make_shared<VecBuffer>(allocator, data, expected_size * sizeof(float), expected_size * sizeof(float), copy);
            else
                buffer_ = std::make_shared<ArrBuffer>(allocator, data, expected_size * sizeof(float), copy);
            update();
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
            VLOG(TRACE) << "Tensor transferred successfully.";
        }

        size_t Tensor::shape(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > shape_.size())
                    return 1;
                return shape_[shape_.size() + idx];
            }
            else
            {
                if (idx >= shape_.size())
                    throw std::out_of_range("Index out of range in shape()");
                return shape_[idx];
            }
        }
        size_t Tensor::stride(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > stride_.size())
                    return size();
                return stride_[stride_.size() + idx];
            }
            else
            {
                if (idx >= stride_.size())
                    throw std::out_of_range("Index out of range in stride()");
                return stride_[idx];
            }
        }

        Tensor Tensor::clone()
        {
            auto new_tensor = Tensor(shape_, buffer_->clone(), device_, mut_);
            new_tensor.stride_ = stride_;
            return new_tensor;
        }

        void Tensor::update()
        {
            if (shape_.empty())
                return;
            if (stride_.back() != 1)
            {
                is_contiguous_ = false;
                return;
            }
            for (int i = static_cast<int>(stride_.size()) - 2; i >= 0; --i)
            {
                if (stride_[i] != stride_[i + 1] * shape_[i + 1])
                {
                    is_contiguous_ = false;
                    return;
                }
            }
            is_contiguous_ = true;
            if (shape_.size() >= 2)
                num_mats_ = size() / (shape(-1) * shape(-2));
            else if (shape_.size() == 1)
                num_mats_ = 1;
            else
                num_mats_ = 0;
        }

        void Tensor::view(std::vector<size_t> shape)
        {
            if (!is_contiguous_)
            {
                throw std::runtime_error("Tensor must be contiguous to use view.");
            }
            size_t new_size = 1;
            for (auto s : shape)
            {
                new_size *= s;
            }
            if (new_size != this->size())
            {
                throw std::invalid_argument("New shape size must match the original size.");
            }
            shape_ = shape;
            stride_ = Tensor::default_stride(shape_);
        }

        void Tensor::reshape(std::vector<size_t> shape)
        {
            if (!is_contiguous_)
            {
                this->contiguous();
            }
            view(shape);
        }

        void Tensor::contiguous(cudaStream_t stream)
        {
            if (is_contiguous_)
                return;
            kernel::get_contiguous_kernel(device_)(this, stream);
            stride_ = default_stride(shape_);
            update();
            CHECK(is_contiguous_) << "Tensor faild to contiguous";
        }
        void Tensor::transpose(int i, int j)
        {
            if (i >= shape_.size() || j >= shape_.size() || -i > shape_.size() || -j > shape_.size())
            {
                throw std::invalid_argument("Invalid transpose dimensions.");
            }
            if (i < 0)
                i += shape_.size();
            if (j < 0)
                j += shape_.size();
            std::swap(shape_[i], shape_[j]);
            std::swap(stride_[i], stride_[j]);
            update();
        }
        void Tensor::t()
        {
            if (shape_.size() < 2)
            {
                throw std::invalid_argument("Invalid transpose dimensions.");
            }
            transpose(shape_.size() - 1, shape_.size() - 2);
        }

        float *Tensor::operator[](size_t idx)
        {
            size_t offset = 0;
            for (int i = shape_.size() - 1; i >= 0; i--)
            {
                offset += (idx % shape_[i]) * stride_[i];
                idx /= shape_[i];
            }
            return this->data() + offset;
        }
        float *Tensor::operator[](std::vector<size_t> idx)
        {
            if (idx.size() != shape_.size())
            {
                throw std::invalid_argument("Index size must match tensor shape size.");
            }
            size_t offset = 0;
            for (size_t i = 0; i < idx.size(); ++i)
            {
                if (idx[i] >= shape_[i])
                {
                    throw std::out_of_range("Index out of range in operator[]");
                }
                offset += idx[i] * stride_[i];
            }
            return data() + offset;
        }

        float *Tensor::mat(size_t idx)
        {
            if (idx >= num_mats_)
            {
                throw std::out_of_range("Matrix index out of range in mat()");
            }
            if (num_mats_ == 1)
                return data();
            size_t offset = 0;
            for (int i = static_cast<int>(stride_.size()) - 3; i >= 0; i--)
            {
                offset += (idx % shape_[i]) * stride_[i];
                idx /= shape_[i];
            }
            return data() + offset;
        }
    } // namespace base
} // namespace mllm