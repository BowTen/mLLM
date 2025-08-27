#include <iostream>
#include <vector>
#include "base/tensor.h"
#include "base/allocator.h"
#include "base/buffer.h"
#include "base/util.h"
#include "kernel/kernel.h"
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {

        void TensorMeta::update()
        {
            if (shape_.size() >= 2)
                num_mats_ = std::accumulate(shape_.begin(), shape_.end() - 2, 1, std::multiplies<size_t>());
            else if (shape_.size() == 1)
                num_mats_ = 1;
            else
                num_mats_ = 0;
            size_ = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());

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
        std::vector<size_t> TensorMeta::default_stride(const std::vector<size_t> &shape)
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
        TensorMeta::TensorMeta() : shape_(),
                                   stride_(default_stride(shape_)),
                                   is_contiguous_(true),
                                   buffer_(nullptr),
                                   device_(Device::CPU),
                                   mut_(false),
                                   stream_(nullptr)
        {
            update();
        }
        TensorMeta::TensorMeta(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device, bool mut, cudaStream_t stream)
            : shape_(shape),
              is_contiguous_(true),
              buffer_(buffer),
              device_(device),
              mut_(mut),
              stream_(stream)
        {
            if (shape_.size() == 1)
                shape_.insert(shape_.begin(), 1);
            stride_ = default_stride(shape_);
            update();
        }

        TensorMeta::TensorMeta(const std::vector<size_t> &shape, Device device, bool mut, cudaStream_t stream)
            : shape_(shape), is_contiguous_(true), device_(device), mut_(mut), stream_(stream)
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

        TensorMeta::TensorMeta(void *data, const std::vector<size_t> &shape, bool copy, Device device, bool mut, cudaStream_t stream)
            : shape_(shape), is_contiguous_(true), device_(device), mut_(mut), stream_(stream)
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

        size_t Tensor::check_index(int idx) const
        {
            if (idx >= 0)
            {
                CHECK(static_cast<size_t>(idx) < meta_->shape_.size()) << "Index out of range, shape size: " << meta_->shape_.size() << ", idx: " << idx;
                return idx;
            }
            else
            {
                CHECK(static_cast<size_t>(-idx) <= meta_->shape_.size()) << "Index out of range, shape size: " << meta_->shape_.size() << ", -idx: " << -idx;
                return meta_->shape_.size() + idx;
            }
        }

        Tensor Tensor::from_float(float value, Device device, bool mut, cudaStream_t stream)
        {
            return Tensor::from_vector(std::vector<float>({value}), {1}, device, mut, stream);
        }

        float *Tensor::data()
        {
            if (!meta_->buffer_)
                return nullptr;
            return static_cast<float *>(meta_->buffer_->data());
        }

        Tensor Tensor::toDevice(Device device)
        {
            if (meta_->device_ == device)
                return *this;
            if (!meta_->buffer_)
            {
                LOG(WARNING) << "Tensor buffer is empty, cannot transfer device.";
                return *this;
            }
            VLOG(4) << "Transferring tensor from device " << (meta_->device_ == Device::CPU ? "CPU" : "CUDA") << " to " << (device == Device::CPU ? "CPU" : "CUDA");

            Allocator *new_allocator = nullptr;
            if (device == Device::CPU)
                new_allocator = HostAllocator::getInstance();
            else if (device == Device::CUDA)
                new_allocator = CudaAllocator::getInstance();
            else
                throw std::invalid_argument("Tensor::toDevice: Unsupported device type.");
            Buffer::BufferPtr new_buffer;
            if (meta_->mut_)
                new_buffer = std::make_shared<VecBuffer>(new_allocator, meta_->buffer_->size() * 2, meta_->buffer_->size());
            else
                new_buffer = std::make_shared<ArrBuffer>(new_allocator, meta_->buffer_->size());
            Allocator::device_memcpy(new_buffer->data(), meta_->buffer_->data(), meta_->buffer_->size(), device == Device::CPU ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice);
            meta_->buffer_ = new_buffer;
            meta_->device_ = device;
            return *this;
        }

        size_t Tensor::shape(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > static_cast<int>(meta_->shape_.size()))
                    return 1;
                return meta_->shape_[meta_->shape_.size() + idx];
            }
            else
            {
                if (idx >= static_cast<int>(meta_->shape_.size()))
                    throw std::out_of_range("Index out of range in shape()");
                return meta_->shape_[idx];
            }
        }
        size_t Tensor::stride(int idx) const
        {
            if (idx < 0)
            {
                if (-idx > static_cast<int>(meta_->stride_.size()))
                    return size();
                return meta_->stride_[meta_->stride_.size() + idx];
            }
            else
            {
                if (idx >= static_cast<int>(meta_->stride_.size()))
                    throw std::out_of_range("Index out of range in stride()");
                return meta_->stride_[idx];
            }
        }

        Tensor Tensor::clone()
        {
            TensorMeta *tm = new TensorMeta(*meta_);
            tm->buffer_ = meta_->buffer_->clone();
            return Tensor(std::shared_ptr<TensorMeta>(tm));
        }

        void Tensor::view(std::vector<size_t> shape)
        {
            size_t new_size = 1;
            size_t dim = meta_->shape_.size();
            for (size_t i = 0; i < dim; i++)
                if (meta_->stride_[i] > 0)
                    new_size *= meta_->shape_[i];

            if (new_size != this->size())
            {
                throw std::invalid_argument("New shape size must match the original size.");
            }
            meta_->shape_ = shape;
            meta_->stride_ = TensorMeta::default_stride(shape);
            meta_->update();
            CHECK(meta_->is_contiguous_) << "Tensor failed to view";
        }

        void Tensor::reshape(std::vector<size_t> shape)
        {
            if (!meta_->is_contiguous_)
            {
                this->contiguous();
            }
            view(shape);
        }

        void Tensor::contiguous()
        {
            if (meta_->is_contiguous_)
                return;
            kernel::get_contiguous_kernel(meta_->device_)(this, this->meta_->stream_);
            meta_->stride_ = TensorMeta::default_stride(meta_->shape_);
            meta_->update();
            CHECK(meta_->is_contiguous_) << "Tensor faild to contiguous";
        }
        void Tensor::transpose(int i, int j)
        {
            if (i >= static_cast<int>(meta_->shape_.size()) ||
                j >= static_cast<int>(meta_->shape_.size()) ||
                -i > static_cast<int>(meta_->shape_.size()) ||
                -j > static_cast<int>(meta_->shape_.size()))
            {
                throw std::invalid_argument("Invalid transpose dimensions. shape dim: " + std::to_string(meta_->shape_.size()));
            }
            if (i < 0)
                i += meta_->shape_.size();
            if (j < 0)
                j += meta_->shape_.size();
            std::swap(meta_->shape_[i], meta_->shape_[j]);
            std::swap(meta_->stride_[i], meta_->stride_[j]);
            meta_->update();
        }
        void Tensor::t()
        {
            if (meta_->shape_.size() < 2)
            {
                throw std::invalid_argument("Invalid transpose dimensions.");
            }
            transpose(meta_->shape_.size() - 1, meta_->shape_.size() - 2);
        }

        float *Tensor::operator[](size_t idx)
        {
            size_t offset = 0;
            for (int i = meta_->shape_.size() - 1; i >= 0; i--)
            {
                offset += (idx % meta_->shape_[i]) * meta_->stride_[i];
                idx /= meta_->shape_[i];
            }
            return this->data() + offset;
        }
        float *Tensor::operator[](std::vector<size_t> idx)
        {
            if (idx.size() != meta_->shape_.size())
            {
                throw std::invalid_argument("Index size must match tensor shape size.");
            }
            size_t offset = 0;
            for (size_t i = 0; i < idx.size(); ++i)
            {
                CHECK(idx[i] < meta_->shape_[i]) << "Index out of range in operator[{}], dim: " +
                                                        std::to_string(i) +
                                                        ", idx: " +
                                                        std::to_string(idx[i]) +
                                                        ", shape: " +
                                                        std::to_string(meta_->shape_[i]);
                offset += idx[i] * meta_->stride_[i];
            }
            return data() + offset;
        }

        float *Tensor::mat(size_t idx)
        {
            if (idx >= meta_->num_mats_)
            {
                throw std::out_of_range("Matrix index out of range in mat()");
            }
            if (meta_->num_mats_ == 1)
                return data();
            size_t offset = 0;
            for (int i = static_cast<int>(meta_->stride_.size()) - 3; i >= 0; i--)
            {
                offset += (idx % meta_->shape_[i]) * meta_->stride_[i];
                idx /= meta_->shape_[i];
            }
            return data() + offset;
        }

        void Tensor::push(float *bytes, size_t num_bytes)
        {
            CHECK(meta_->mut_) << "Tensor must be mutable in push()";
            CHECK(bytes != nullptr) << "Invalid data pointer in push()";
            CHECK(meta_->buffer_ != nullptr) << "Tensor buffer is null in push()";

            auto vec_buffer = std::dynamic_pointer_cast<VecBuffer>(meta_->buffer_);
            CHECK(vec_buffer != nullptr) << "Tensor buffer is not a VecBuffer in push()";

            vec_buffer->push(bytes, num_bytes);
        }

        void Tensor::cat(Tensor &other, int dim_int)
        {
            CHECK(meta_->mut_) << "Tensor must be mutable in cat()";
            int dim = check_index(dim_int);
            CHECK(other.shape().size() == this->shape().size()) << "Other tensor must have the same number of dimensions.";
            auto other_shape = other.shape();
            other_shape[dim] = meta_->shape_[dim];
            CHECK(other_shape == meta_->shape_) << "Other tensor must have the same shape except for the concatenation dimension.";

            std::vector<size_t> next_idx(meta_->shape_.size(), 0);
            next_idx[dim] = meta_->shape_[dim] - 1;
            float *next_ptr = this->operator[](next_idx) + meta_->stride_[dim];
            if (next_ptr == this->data() + this->size() && other.stride() == this->stride())
            {
                VLOG(DEBUG) << "directly pushing data to cat Tensors";
                this->push(other.data(), other.size() * sizeof(float));
                this->meta_->shape_[dim] += other.shape(dim);
                this->view(this->shape());
                return;
            }

            LOG(WARNING) << "Falling back to cloning tensors for cat()";
            Tensor this_cp = this->clone();
            size_t total_size = this_cp.logic_size() + other.logic_size();
            this->resize(total_size);

            this_cp.contiguous();
            other.contiguous();
            size_t block_size = std::accumulate(meta_->shape_.begin() + dim + 1, meta_->shape_.end(), 1, std::multiplies<size_t>());

            float *new_data = this->data();
            float *new_data_end = new_data + total_size;
            float *this_cp_data = this_cp.data();
            float *other_data = other.data();

            size_t this_cp_stride = this_cp.shape(dim) * block_size;
            size_t other_stride = other.shape(dim) * block_size;
            Allocator *allocator = this->meta_->buffer_->get_allocator();
            while (new_data < new_data_end)
            {
                allocator->memcpy(new_data, this_cp_data, this_cp_stride * sizeof(float));
                new_data += this_cp_stride;
                this_cp_data += this_cp_stride;
                allocator->memcpy(new_data, other_data, other_stride * sizeof(float));
                new_data += other_stride;
                other_data += other_stride;
            }

            this->meta_->shape_[dim] += other.shape(dim);
            this->view(this->shape());
        }

        void Tensor::insert_dim(int dim_int)
        {
            size_t dim = check_index(dim_int);
            CHECK(dim < meta_->shape_.size()) << "Dimension out of range in insert_dim()";
            meta_->shape_.insert(meta_->shape_.begin() + dim, 1);
            meta_->stride_.insert(meta_->stride_.begin() + dim, 0);
            meta_->stride_[dim] = meta_->stride_[dim + 1] * meta_->shape_[dim + 1];
            meta_->update();
        }
        void Tensor::expand(int dim_int, size_t size)
        {
            size_t dim = check_index(dim_int);
            CHECK(meta_->shape_[dim] == 1) << "Dimension must be 1 in expand()";
            meta_->shape_[dim] = size;
            meta_->stride_[dim] = 0;
            meta_->update();
        }
        void Tensor::reserve(size_t size)
        {
            CHECK(meta_->mut_) << "Tensor must be mutable in reserve()";
            std::shared_ptr<VecBuffer> vec_buffer = std::dynamic_pointer_cast<VecBuffer>(meta_->buffer_);
            CHECK(vec_buffer);
            vec_buffer->reserve(size * sizeof(float));
        }
        void Tensor::resize(size_t size)
        {
            CHECK(meta_->mut_) << "Tensor must be mutable in reserve()";
            std::shared_ptr<VecBuffer> vec_buffer = std::dynamic_pointer_cast<VecBuffer>(meta_->buffer_);
            CHECK(vec_buffer);
            vec_buffer->resize(size * sizeof(float));
        }

        std::vector<size_t> Tensor::index(size_t id)
        {
            std::vector<size_t> idx(meta_->shape_.size());
            for (int i = meta_->shape_.size() - 1; i >= 0; --i)
            {
                idx[i] = id % meta_->shape_[i];
                id /= meta_->shape_[i];
            }
            return idx;
        }

    } // namespace base
} // namespace mllm