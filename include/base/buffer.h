#ifndef MLLM_BUFFER_H
#define MLLM_BUFFER_H

#include <cstddef>

namespace mllm
{
    namespace base
    {
        class Allocator;

        class Buffer
        {
        protected:
            Allocator *allocator;
            void *data_;

        public:
            Buffer(Allocator *alloc, size_t size);
            virtual ~Buffer();

            virtual size_t size() const = 0;

            void *data();
        };

        class ArrBuffer : public Buffer
        {
            size_t size_;

        public:
            ArrBuffer(Allocator *alloc, size_t size);

            size_t size() const override;
        };

        class VecBuffer : public Buffer
        {
            size_t size_;
            size_t capacity_;

        public:
            VecBuffer(Allocator *alloc, size_t initial_capacity, size_t initial_size);

            void concat(const void *bytes, size_t num_bytes);

            void reserve(size_t new_capacity);

            size_t size() const override;

            size_t capacity() const;
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BUFFER_H
