#include "base/buffer.h"
#include "base/allocator.h"
#include "base/common.h"
#include <iostream>
#include <algorithm>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        // Buffer implementation
        Buffer::Buffer(Allocator *alloc, size_t size)
            : allocator(alloc)
        {
            if (size == 0)
            {
                VLOG(DEBUG) << "allocating zero size buffer, data pointer will be null.";
                data_ = nullptr;
                return;
            }
            data_ = allocator->allocate(size);
            if (!data_)
            {
                throw std::bad_alloc();
            }
        }
        Buffer::Buffer(Allocator *alloc, void *data)
            : allocator(alloc), data_(data)
        {
        }

        Buffer::~Buffer()
        {
            if (data_)
            {
                allocator->deallocate(data_);
            }
        }

        void *Buffer::data()
        {
            return data_;
        }

        // ArrBuffer implementation
        ArrBuffer::ArrBuffer(Allocator *alloc, size_t size)
            : Buffer(alloc, size), size_(size)
        {
        }
        ArrBuffer::ArrBuffer(Allocator *alloc, void *data, size_t size)
            : Buffer(alloc, data), size_(size)
        {
            if (!data_ && size > 0)
            {
                LOG(ERROR) << "ArrBuffer initialized with null data pointer but size is greater than zero.";
            }
        }

        size_t ArrBuffer::size() const
        {
            return size_;
        }

        // VecBuffer implementation
        VecBuffer::VecBuffer(Allocator *alloc, size_t initial_capacity, size_t initial_size)
            : Buffer(alloc, std::max(initial_size, initial_capacity)),
              size_(initial_size),
              capacity_(std::max(initial_capacity, initial_size))
        {
        }
        VecBuffer::VecBuffer(Allocator *alloc, void *data, size_t initial_capacity, size_t initial_size)
            : Buffer(alloc, data),
              size_(initial_size),
              capacity_(std::max(initial_capacity, initial_size))
        {
            if (!data_ && initial_size > 0)
            {
                LOG(ERROR) << "VecBuffer initialized with null data pointer but size is greater than zero.";
            }
        }

        void VecBuffer::concat(const void *bytes, size_t num_bytes)
        {
            if (size_ + num_bytes > capacity_)
            {
                reserve(std::max(capacity_ * 2, size_ + num_bytes));
            }
            allocator->memcpy(static_cast<char *>(data_) + size_, bytes, num_bytes);
            size_ += num_bytes;
        }

        void VecBuffer::reserve(size_t new_capacity)
        {
            if (new_capacity > capacity_)
            {
                void *new_data = allocator->allocate(new_capacity);
                if (!new_data)
                {
                    throw std::bad_alloc();
                }
                allocator->memcpy(new_data, data_, size_);
                allocator->deallocate(data_);
                data_ = new_data;
                capacity_ = new_capacity;
            }
        }

        size_t VecBuffer::size() const
        {
            return size_;
        }

        size_t VecBuffer::capacity() const
        {
            return capacity_;
        }
    } // namespace base
} // namespace mllm
