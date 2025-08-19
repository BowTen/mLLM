#ifndef MLLM_BASE_TENSOR_H
#define MLLM_BASE_TENSOR_H

#include <iostream>
#include <vector>
#include <armadillo>
#include "allocator.h"
#include "buffer.h"
#include "common.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        class Tensor
        {
            std::vector<size_t> shape_;
            // std::vector<size_t> stride_;
            Buffer::BufferPtr buffer_;
            Device device_;
            bool mut_;

        public:
            Tensor() : shape_(),
                       buffer_(nullptr),
                       device_(Device::CPU) {}
            Tensor(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device = Device::CPU, bool mut = false)
                : shape_(shape), buffer_(buffer), device_(device), mut_(mut) {}
            Tensor(const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);
            Tensor(void *data, const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);

            const std::vector<size_t> &shape() const;
            size_t size() const;
            float *data();
            bool empty() const { return buffer_ == nullptr; }
            void toDevice(Device device);
            Tensor clone();
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_TENSOR_H