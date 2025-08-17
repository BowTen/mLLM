#ifndef MLLM_ALLOCATOR_H
#define MLLM_ALLOCATOR_H

#include <cstddef>

namespace mllm
{
    namespace base
    {
        class Allocator
        {
        public:
            Allocator() = default;
            virtual ~Allocator() = default;

            // 分配内存
            virtual void *allocate(size_t size) = 0;

            // 复制内存
            virtual void memcpy(void *dest, const void *src, size_t size) = 0;

            // 释放内存
            virtual void deallocate(void *ptr) = 0;
        };

        class HostAllocator : public Allocator
        {
            static HostAllocator *instance;
            HostAllocator() = default;

        public:
            ~HostAllocator() override = default;

            static HostAllocator *getInstance();

            void *allocate(size_t size) override;

            void memcpy(void *dest, const void *src, size_t size) override;

            void deallocate(void *ptr) override;
        };

        class CudaAllocator : public Allocator
        {
            static CudaAllocator *instance;
            CudaAllocator() = default;

        public:
            ~CudaAllocator() override = default;

            static CudaAllocator *getInstance();

            void *allocate(size_t size) override;

            void memcpy(void *dest, const void *src, size_t size) override;

            void deallocate(void *ptr) override;
        };
    }
}

#endif // MLLM_ALLOCATOR_H
