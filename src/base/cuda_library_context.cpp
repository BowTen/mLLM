#include "base/cuda_library_context.h"

namespace mllm
{
    namespace base
    {
        CudaLibraryContext::CudaLibraryContext()
        {
            CHECK_CUBLAS_ERR(cublasCreate(&cublas_handle_));
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            CHECK_CUDNN_ERR(cudnnCreate(&cudnn_handle_));
#endif
        }

        CudaLibraryContext::~CudaLibraryContext()
        {
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            if (cudnn_handle_ != nullptr)
            {
                cudnnStatus_t status = cudnnDestroy(cudnn_handle_);
                if (status != CUDNN_STATUS_SUCCESS)
                {
                    LOG(ERROR) << "cuDNN error during handle teardown: "
                               << cudnn_status_string(status);
                }
                cudnn_handle_ = nullptr;
            }
#endif
            if (cublas_handle_ != nullptr)
            {
                cublasStatus_t status = cublasDestroy(cublas_handle_);
                if (status != CUBLAS_STATUS_SUCCESS)
                {
                    LOG(ERROR) << "cuBLAS error during handle teardown: "
                               << cublas_status_string(status);
                }
                cublas_handle_ = nullptr;
            }
        }

        void CudaLibraryContext::bind_stream(cudaStream_t stream)
        {
            stream_ = stream;
            CHECK_CUBLAS_ERR(cublasSetStream(cublas_handle_, stream_));
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
            CHECK_CUDNN_ERR(cudnnSetStream(cudnn_handle_, stream_));
#endif
        }
    }
}
