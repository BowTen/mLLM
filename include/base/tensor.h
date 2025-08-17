#ifndef MLLM_BASE_TENSOR_H
#define MLLM_BASE_TENSOR_H

#include <iostream>
#include <vector>
#include "allocator.h"
#include "buffer.h"

namespace mllm
{
    namespace base
    {
        enum Device
        {
            CPU = 0,
            CUDA = 1,
        };

        class Tensor
        {
            std::vector<size_t> shape_;
            Buffer::BufferPtr buffer_;
            Device device_;

        public:
            Tensor() : shape_(),
                       buffer_(nullptr),
                       device_(Device::CPU) {}
            Tensor(const std::vector<size_t> &shape, Device device = Device::CPU, bool mut = false);

            const std::vector<size_t> &shape() const;
            size_t size() const;
            float *data();
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_TENSOR_H