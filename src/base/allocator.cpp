#include "base/allocator.h"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

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

#define BYTE_ALIGN 16
        void *CudaAllocator::allocate(size_t size)
        {
            CHECK(size > 0) << "Cannot allocate zero bytes";
            size = ((size + BYTE_ALIGN - 1) / BYTE_ALIGN) * BYTE_ALIGN;
            void *ptr;
            auto err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess)
            {
                LOG(FATAL) << "Cuda allocation failed while alloce " << size << " bytes: " << cudaGetErrorString(err);
                throw std::bad_alloc();
            }
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
