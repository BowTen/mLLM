#ifndef MLLM_BASE_CUDA_LIBRARY_ERROR_H
#define MLLM_BASE_CUDA_LIBRARY_ERROR_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
#include <cudnn.h>
#endif

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        inline const char *cublas_status_string(cublasStatus_t status)
        {
            switch (status)
            {
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
            default:
                return "CUBLAS_STATUS_UNKNOWN";
            }
        }

#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
        inline const char *cudnn_status_string(cudnnStatus_t status)
        {
            return cudnnGetErrorString(status);
        }
#endif
    }
}

#define CHECK_CUBLAS_ERR(err)                                                 \
    do                                                                        \
    {                                                                         \
        cublasStatus_t status = (err);                                        \
        if (status != CUBLAS_STATUS_SUCCESS)                                  \
        {                                                                     \
            CHECK(false) << "cuBLAS error: "                                 \
                         << mllm::base::cublas_status_string(status);        \
        }                                                                     \
    } while (0)

#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
#define CHECK_CUDNN_ERR(err)                                                  \
    do                                                                        \
    {                                                                         \
        cudnnStatus_t status = (err);                                         \
        if (status != CUDNN_STATUS_SUCCESS)                                   \
        {                                                                     \
            CHECK(false) << "cuDNN error: "                                   \
                         << mllm::base::cudnn_status_string(status);           \
        }                                                                     \
    } while (0)
#endif

#endif // MLLM_BASE_CUDA_LIBRARY_ERROR_H
