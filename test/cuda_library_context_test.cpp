#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
#include <cudnn.h>
#endif

#include "base/cuda_library_context.h"

using namespace mllm::base;

TEST(CudaLibraryContextTest, StartsWithDefaultStream)
{
    CudaLibraryContext context;
    ASSERT_NE(context.cublas_handle(), nullptr);
    EXPECT_EQ(context.stream(), nullptr);

    cudaStream_t bound_stream = reinterpret_cast<cudaStream_t>(0x1);
    ASSERT_EQ(cublasGetStream(context.cublas_handle(), &bound_stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, nullptr);
}

TEST(CudaLibraryContextTest, RebindsStreamToCublasHandle)
{
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream2), cudaSuccess);

    CudaLibraryContext context;
    ASSERT_NE(context.cublas_handle(), nullptr);

    context.bind_stream(stream1);
    cudaStream_t bound_stream = nullptr;
    ASSERT_EQ(cublasGetStream(context.cublas_handle(), &bound_stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, stream1);

    context.bind_stream(stream2);
    ASSERT_EQ(cublasGetStream(context.cublas_handle(), &bound_stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, stream2);

    context.bind_stream(nullptr);
    ASSERT_EQ(cublasGetStream(context.cublas_handle(), &bound_stream), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, nullptr);

    ASSERT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream2), cudaSuccess);
}

#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
TEST(CudaLibraryContextTest, RebindsStreamToCudnnHandle)
{
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream2), cudaSuccess);

    CudaLibraryContext context;
    ASSERT_NE(context.cudnn_handle(), nullptr);

    context.bind_stream(stream1);
    cudaStream_t bound_stream = nullptr;
    ASSERT_EQ(cudnnGetStream(context.cudnn_handle(), &bound_stream), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, stream1);

    context.bind_stream(stream2);
    ASSERT_EQ(cudnnGetStream(context.cudnn_handle(), &bound_stream), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, stream2);

    context.bind_stream(nullptr);
    ASSERT_EQ(cudnnGetStream(context.cudnn_handle(), &bound_stream), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(bound_stream, nullptr);

    ASSERT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream2), cudaSuccess);
}
#endif
