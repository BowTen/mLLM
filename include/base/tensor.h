#ifndef MLLM_BASE_TENSOR_H
#define MLLM_BASE_TENSOR_H

#include <iostream>
#include <vector>
#include <armadillo>
#include "allocator.h"
#include "buffer.h"
#include "common.h"
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        class Tensor;

        class TensorMeta
        {
        private:
            std::vector<size_t> shape_;
            std::vector<size_t> stride_;
            bool is_contiguous_;
            Buffer::BufferPtr buffer_;
            Buffer::BufferPtr buf_holder_;
            Device device_;
            bool mut_;
            size_t num_mats_ = 0;

            static std::vector<size_t> default_stride(const std::vector<size_t> &shape);
            void update();

            friend Tensor;

        public:
            TensorMeta();
            TensorMeta(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device = Device::CPU, bool mut = false);
            TensorMeta(const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);
            TensorMeta(void *data, const std::vector<size_t> &shape, bool copy, Device device = Device::CPU, bool mut = false);
        };

        class Tensor
        {
        private:
            std::shared_ptr<TensorMeta> meta_;

            size_t check_index(int idx) const;

        public:
            Tensor() : meta_(std::make_shared<TensorMeta>()) {}
            Tensor(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device = Device::CPU, bool mut = false)
                : meta_(std::make_shared<TensorMeta>(shape, buffer, device, mut)) {}
            Tensor(const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false)
                : meta_(std::make_shared<TensorMeta>(shape, device, mut)) {}
            Tensor(void *data, const std::vector<size_t> &shape, bool copy, Device device = Device::CPU, bool mut = false)
                : meta_(std::make_shared<TensorMeta>(data, shape, copy, device, mut)) {}
            static Tensor from_float(float value, Device device = Device::CPU, bool mut = false);

            template <class T>
            static Tensor from_vector(std::vector<T> vec, std::vector<size_t> shape, Device device = Device::CPU, bool mut = false)
            {
                Tensor tensor(vec.data(), shape, true, Device::CPU, mut);
                tensor.toDevice(device);
                return tensor;
            }

            void view(std::vector<size_t> shape);
            void reshape(std::vector<size_t> shape);
            void contiguous(cudaStream_t stream = nullptr);
            void transpose(int i, int j);
            void t();

            float *operator[](size_t idx);
            float *operator[](std::vector<size_t> idx);
            size_t num_mats()
            {
                meta_->update();
                return meta_->num_mats_;
            }
            float *mat(size_t idx);

            const std::vector<size_t> &shape() const { return meta_->shape_; }
            size_t shape(int idx) const;
            size_t stride(int idx) const;
            const std::vector<size_t> &stride() const { return meta_->stride_; }
            void set_buffer(Buffer::BufferPtr buffer) { meta_->buffer_ = buffer; }
            Buffer::BufferPtr buffer() const { return meta_->buffer_; }
            bool is_contiguous() const { return meta_->is_contiguous_; }
            size_t size() const;
            float *data();
            bool empty() const { return meta_->buffer_ == nullptr; }
            Tensor toDevice(Device device);
            Tensor clone();

            void push(float *bytes, size_t num_bytes);
            void cat(Tensor &other, int dim_int);

            void insert_dim(int dim_int);
            void expand(int dim_int, size_t size);
        };

        using PosEmb = std::pair<Tensor *, Tensor *>; // cos sin
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_TENSOR_H