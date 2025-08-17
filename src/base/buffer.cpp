#include "base/buffer.h"
#include "base/allocator.h"
#include <iostream>
#include <algorithm>

namespace mllm
{
    namespace base
    {
        // Buffer implementation
        Buffer::Buffer(Allocator *alloc, size_t size)
            : allocator(alloc), data_(allocator->allocate(size))
        {
            if (!data_)
            {
                throw std::bad_alloc();
            }
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
