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
        class Tensor
        {
            std::vector<size_t> shape_;
            std::vector<size_t> stride_;
            bool is_contiguous_;
            Buffer::BufferPtr buffer_;
            Device device_;
            bool mut_;

            static std::vector<size_t> default_stride(const std::vector<size_t> &shape);
            void check_contiguous();

        public:
            Tensor() : shape_(),
                       stride_(default_stride(shape_)),
                       is_contiguous_(true),
                       buffer_(nullptr),
                       device_(Device::CPU) {}
            Tensor(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device = Device::CPU, bool mut = false)
                : shape_(shape),
                  stride_(default_stride(shape)),
                  is_contiguous_(true),
                  buffer_(buffer),
                  device_(device),
                  mut_(mut) {}
            Tensor(const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);
            Tensor(void *data, const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);

            void view(std::vector<size_t> shape);
            void reshape(std::vector<size_t> shape);
            void contiguous(cudaStream_t stream = nullptr);
            void transpose(size_t i, size_t j);
            void t();

            const std::vector<size_t> &shape() const { return shape_; }
            const std::vector<size_t> &stride() const { return stride_; }
            void set_buffer(Buffer::BufferPtr buffer) { buffer_ = buffer; }
            Buffer::BufferPtr buffer() const { return buffer_; }
            bool is_contiguous() const { return is_contiguous_; }
            size_t size() const;
            float *data();
            bool empty() const { return buffer_ == nullptr; }
            void toDevice(Device device);
            Tensor clone();
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_TENSOR_H