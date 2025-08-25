#ifndef MLLM_BUFFER_H
#define MLLM_BUFFER_H

#include <cstddef>
#include <memory>

namespace mllm
{
    namespace base
    {
        class Allocator;

        // TODO: 目前Tensor::toDevice方法会直接重构一个新的buffer，这会导致与其他共享Tensor的数据不一致，可以直接修改buffer数据来解决
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
            virtual BufferPtr clone(bool copy_data = true) const = 0;

            void *data();
            Allocator *get_allocator() const { return allocator; }
        };

        class ArrBuffer : public Buffer
        {
            size_t size_;

        public:
            ArrBuffer(Allocator *alloc, size_t size);
            ArrBuffer(Allocator *alloc, void *data, size_t size, bool copy = true);

            size_t size() const override;
            BufferPtr clone(bool copy_data = true) const override;
        };

        class VecBuffer : public Buffer
        {
            size_t size_;
            size_t capacity_;

        public:
            VecBuffer(Allocator *alloc, size_t initial_capacity, size_t initial_size);
            VecBuffer(Allocator *alloc, void *data, size_t initial_capacity, size_t initial_size, bool copy = true);

            void push(const void *bytes, size_t num_bytes);

            void reserve(size_t new_capacity);
            void resize(size_t new_size);

            size_t size() const override;
            BufferPtr clone(bool copy_data = true) const override;

            size_t capacity() const;
        };
    } // namespace base
} // namespace mllm

#endif // MLLM_BUFFER_H
