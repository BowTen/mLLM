#include "base/allocator.h"
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>
#include "base/util.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {

        size_t align_size(size_t size)
        {
            return ((size + MEM_ALIGN - 1) / MEM_ALIGN) * MEM_ALIGN;
        }

        void Allocator::device_memcpy(void *dest, const void *src, size_t size, cudaMemcpyKind kind)
        {
            CHECK_CUDA_ERR(cudaMemcpy(dest, src, size, kind));
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
        }

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
            size = align_size(size);
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
            CHECK(size > 0) << "Cannot allocate zero bytes";
            size = align_size(size);
            void *ptr;
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
            CHECK_CUDA_ERR(cudaMalloc(&ptr, size));
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
            return ptr;
        }

        void CudaAllocator::memcpy(void *dest, const void *src, size_t size)
        {
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
            CHECK_CUDA_ERR(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
        }

        void CudaAllocator::deallocate(void *ptr)
        {
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
            CHECK_CUDA_ERR(cudaFree(ptr));
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());
        }
    }
}
