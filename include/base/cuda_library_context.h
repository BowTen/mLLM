#ifndef MLLM_BASE_CUDA_LIBRARY_CONTEXT_H
#define MLLM_BASE_CUDA_LIBRARY_CONTEXT_H

#include "base/cuda_library_error.h"

namespace mllm
{
    namespace base
    {
        class CudaLibraryContext
        {
        public:
            CudaLibraryContext();
            ~CudaLibraryContext();

            CudaLibraryContext(const CudaLibraryContext &) = delete;
            CudaLibraryContext &operator=(const CudaLibraryContext &) = delete;
            CudaLibraryContext(CudaLibraryContext &&) = delete;
            CudaLibraryContext &operator=(CudaLibraryContext &&) = delete;

            cublasHandle_t cublas_handle() const { return cublas_handle_; }
            cudaStream_t stream() const { return stream_; }
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            cudnnHandle_t cudnn_handle() const { return cudnn_handle_; }
#endif

            void bind_stream(cudaStream_t stream);

        private:
            cublasHandle_t cublas_handle_ = nullptr;
            cudaStream_t stream_ = nullptr;
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            cudnnHandle_t cudnn_handle_ = nullptr;
#endif
        };
    }
}

#endif // MLLM_BASE_CUDA_LIBRARY_CONTEXT_H
