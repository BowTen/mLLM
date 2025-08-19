#ifndef MLLM_BUFFER_H
#define MLLM_BUFFER_H

#include <cstddef>
#include <memory>

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
            using BufferPtr = std::shared_ptr<Buffer>;

            Buffer(Allocator *alloc, size_t size);
            Buffer(Allocator *alloc, void *data);
            virtual ~Buffer();

            virtual size_t size() const = 0;

            void *data();
        };

        class ArrBuffer : public Buffer
        {
            size_t size_;

        public:
            ArrBuffer(Allocator *alloc, size_t size);
            ArrBuffer(Allocator *alloc, void *data, size_t size);

            size_t size() const override;
        };

        class VecBuffer : public Buffer
        {
            size_t size_;
            size_t capacity_;

        public:
            VecBuffer(Allocator *alloc, size_t initial_capacity, size_t initial_size);
            VecBuffer(Allocator *alloc, void *data, size_t initial_capacity, size_t initial_size);

            void concat(const void *bytes, size_t num_bytes);

            void reserve(size_t new_capacity);

            size_t size() const override;

            size_t capacity() const;
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BUFFER_H
