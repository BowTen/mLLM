#include "base/allocator.h"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

namespace mllm
{
    namespace base
    {
        // HostAllocator implementation
        HostAllocator *HostAllocator::instance = nullptr;

        HostAllocator *HostAllocator::getInstance()
        {
            if (!instance)
            {
                instance = new HostAllocator();
            }
            return instance;
        }

        void *HostAllocator::allocate(size_t size)
        {
            return malloc(size);
        }

        void HostAllocator::memcpy(void *dest, const void *src, size_t size)
        {
            std::memcpy(dest, src, size);
        }

        void HostAllocator::deallocate(void *ptr)
        {
            free(ptr);
        }

        // CudaAllocator implementation
        CudaAllocator *CudaAllocator::instance = nullptr;

        CudaAllocator *CudaAllocator::getInstance()
        {
            if (!instance)
            {
                instance = new CudaAllocator();
            }
            return instance;
        }

        void *CudaAllocator::allocate(size_t size)
        {
            void *ptr;
            cudaMalloc(&ptr, size);
            return ptr;
        }

        void CudaAllocator::memcpy(void *dest, const void *src, size_t size)
        {
            cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
        }

        void CudaAllocator::deallocate(void *ptr)
        {
            cudaFree(ptr);
        }
    }
}
