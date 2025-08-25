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

        size_t Tensor::check_index(int idx) const
        {
            if (idx >= 0)
            {
                CHECK(static_cast<size_t>(idx) < shape_.size()) << "Index out of range, shape size: " << shape_.size() << ", idx: " << idx;
                return idx;
            }
            else
            {
                CHECK(static_cast<size_t>(-idx) <= shape_.size()) << "Index out of range, shape size: " << shape_.size() << ", -idx: " << -idx;
                return shape_.size() + idx;
            }
        }

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
                throw std::invalid_argument("Tensor::Tensor(const std::vector<size_t> &shape, Device device, bool mut): Unsupported device type.");
            if (mut)
            {
                buffer_ = std::make_shared<VecBuffer>(allocator, expected_size * sizeof(float) * 2, expected_size * sizeof(float));
            }
            else
            {
                CHECK(expected_size > 0) << "Non-Mutable Tensor size must be greater than 0";
                buffer_ = std::make_shared<ArrBuffer>(allocator, expected_size * sizeof(float));
            }
            update();
        }

        Tensor::Tensor(void *data, const std::vector<size_t> &shape, bool copy, Device device, bool mut)
            : shape_(shape), is_contiguous_(true), device_(device), mut_(mut)
        {
            if (shape_.size() == 1)
                shape_.insert(shape_.begin(), 1);
            stride_ = default_stride(shape_);
            CHECK(isDevicePointer(data) == (device == Device::CUDA)) << "Data pointer device type does not match Tensor device type.";
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
                throw std::invalid_argument("Tensor::Tensor(void *data... ):Unsupported device type.");
            if (mut)
                buffer_ = std::make_shared<VecBuffer>(allocator, data, expected_size * sizeof(float), expected_size * sizeof(float), copy);
            else
                buffer_ = std::make_shared<ArrBuffer>(allocator, data, expected_size * sizeof(float), copy);
            update();
        }
        Tensor Tensor::from_float(float value, Device device, bool mut)
        {
            return Tensor::from_vector(std::vector<float>({value}), {1}, device, mut);
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

        Tensor *Tensor::toDevice(Device device)
        {
            if (device_ == device)
                return this;
            if (!buffer_)
            {
                LOG(WARNING) << "Tensor buffer is empty, cannot transfer device.";
                return this;
            }
            VLOG(4) << "Transferring tensor from device " << (device_ == Device::CPU ? "CPU" : "CUDA") << " to " << (device == Device::CPU ? "CPU" : "CUDA");

            Allocator *new_allocator = nullptr;
            if (device == Device::CPU)
                new_allocator = HostAllocator::getInstance();
            else if (device == Device::CUDA)
                new_allocator = CudaAllocator::getInstance();
            else
                throw std::invalid_argument("Tensor::toDevice: Unsupported device type.");
            Buffer::BufferPtr new_buffer;
            if (mut_)
                new_buffer = std::make_shared<VecBuffer>(new_allocator, buffer_->size() * 2, buffer_->size());
            else
                new_buffer = std::make_shared<ArrBuffer>(new_allocator, buffer_->size());
            cudaMemcpy(new_buffer->data(), buffer_->data(), buffer_->size(), device == Device::CPU ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
            buffer_ = new_buffer;
            device_ = device;
            return this;
        }

        size_t Tensor::shape(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > static_cast<int>(shape_.size()))
                    return 1;
                return shape_[shape_.size() + idx];
            }
            else
            {
                if (idx >= static_cast<int>(shape_.size()))
                    throw std::out_of_range("Index out of range in shape()");
                return shape_[idx];
            }
        }
        size_t Tensor::stride(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > static_cast<int>(stride_.size()))
                    return size();
                return stride_[stride_.size() + idx];
            }
            else
            {
                if (idx >= static_cast<int>(stride_.size()))
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
            if (shape_.size() >= 2)
                num_mats_ = std::accumulate(shape_.begin(), shape_.end() - 2, 1, std::multiplies<size_t>());
            else if (shape_.size() == 1)
                num_mats_ = 1;
            else
                num_mats_ = 0;
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
        }

        void Tensor::view(std::vector<size_t> shape)
        {
            if (!is_contiguous_)
            {
                throw std::runtime_error("Tensor must be contiguous to use view.");
            }
            size_t new_size = 1;
            size_t dim = shape_.size();
            for (size_t i = 0; i < dim; i++)
                if (stride_[i] > 0)
                    new_size *= shape_[i];

            if (new_size != this->size())
            {
                throw std::invalid_argument("New shape size must match the original size.");
            }
            shape_ = shape;
            stride_ = Tensor::default_stride(shape_);
            update();
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
            if (i >= static_cast<int>(shape_.size()) ||
                j >= static_cast<int>(shape_.size()) ||
                -i > static_cast<int>(shape_.size()) ||
                -j > static_cast<int>(shape_.size()))
            {
                throw std::invalid_argument("Invalid transpose dimensions. shape dim: " + std::to_string(shape_.size()));
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
                CHECK(idx[i] < shape_[i]) << "Index out of range in operator[{}], dim: " +
                                                 std::to_string(i) +
                                                 ", idx: " +
                                                 std::to_string(idx[i]) +
                                                 ", shape: " +
                                                 std::to_string(shape_[i]);
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

        void Tensor::push(float *bytes, size_t num_bytes)
        {
            CHECK(mut_) << "Tensor must be mutable in push()";
            CHECK(bytes != nullptr) << "Invalid data pointer in push()";
            CHECK(buffer_ != nullptr) << "Tensor buffer is null in push()";

            auto vec_buffer = std::dynamic_pointer_cast<VecBuffer>(buffer_);
            CHECK(vec_buffer != nullptr) << "Tensor buffer is not a VecBuffer in push()";

            vec_buffer->push(bytes, num_bytes);
        }

        void Tensor::cat(Tensor &other, int dim)
        {
            CHECK(mut_) << "Tensor must be mutable in cat()";
            CHECK(dim < static_cast<int>(shape_.size()) && -dim <= static_cast<int>(shape_.size())) << "Dimension out of range in cat()";
            CHECK(other.shape().size() == this->shape().size()) << "Other tensor must have the same number of dimensions.";
            if (dim < 0)
                dim += static_cast<int>(shape_.size());
            auto other_shape = other.shape();
            other_shape[dim] = this->shape_[dim];
            CHECK(other_shape == this->shape_) << "Other tensor must have the same shape except for the concatenation dimension.";

            std::vector<size_t> next_idx(shape_.size(), 0);
            next_idx[dim] = this->shape_[dim] - 1;
            float *next_ptr = this->operator[](next_idx) + stride_[dim];
            if (next_ptr == this->data() + this->size() && other.stride() == this->stride())
            {
                VLOG(DEBUG) << "directly pushing data to cat Tensors";
                this->push(other.data(), other.size() * sizeof(float));
                this->shape_[dim] += other.shape(dim);
                update();
                return;
            }

            LOG(WARNING) << "Falling back to cloning tensors for cat()";
            auto cp = this->clone();
            cp.contiguous();
            other.contiguous();
            buf_holder_ = cp.buffer();

            auto vec_buffer = std::dynamic_pointer_cast<VecBuffer>(buffer_);
            CHECK(vec_buffer != nullptr) << "Buffer is not a VecBuffer in cat()";

            vec_buffer->resize((this->size() + other.size()) * sizeof(float));

            size_t dim_size = std::accumulate(shape_.begin() + dim + 1, shape_.end(), 1, std::multiplies<size_t>());
            size_t this_num_dims = this->size() / dim_size;
            size_t other_num_dims = other.size() / dim_size;

            auto allocator = this->buffer_->get_allocator();
            float *st = this->data();
            float *ed = st + this->size();
            float *src_a = cp.data();
            float *src_b = other.data();
            size_t stride_a = dim_size * this_num_dims;
            size_t stride_b = dim_size * other_num_dims;
            size_t num_a_copy_bytes = stride_a * sizeof(float);
            size_t num_b_copy_bytes = stride_b * sizeof(float);
            while (st < ed)
            {
                allocator->memcpy(st, src_a, num_a_copy_bytes);
                st += stride_a;
                src_a += stride_a;
                allocator->memcpy(st, src_b, num_b_copy_bytes);
                st += stride_b;
                src_b += stride_b;
            }
            shape_[dim] += other.shape(dim);
            update();
        }

        void Tensor::insert_dim(int dim_int)
        {
            size_t dim = check_index(dim_int);
            CHECK(dim < shape_.size()) << "Dimension out of range in insert_dim()";
            shape_.insert(shape_.begin() + dim, 1);
            stride_.insert(stride_.begin() + dim, 0);
            stride_[dim] = stride_[dim + 1] * shape_[dim + 1];
            update();
        }
        void Tensor::expand(int dim_int, size_t size)
        {
            size_t dim = check_index(dim_int);
            CHECK(shape_[dim] == 1) << "Dimension must be 1 in expand()";
            shape_[dim] = size;
            stride_[dim] = 0;
            update();
        }
    } // namespace base
} // namespace mllm