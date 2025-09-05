#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel/cuda/gemm_kernel.cuh"
#include <iostream>
#include <fstream>

using namespace mllm::base;

#define CHECK_CUDA_ERR(call)                                              \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }
#define CHECK_CUBLAS_ERR(call)                                                           \
    {                                                                                    \
        cublasStatus_t status = call;                                                    \
        if (status != CUBLAS_STATUS_SUCCESS)                                             \
        {                                                                                \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

const int NUM_ROUND = 500;

// ret: seconds
float run_my_gemm(Tensor A, Tensor B, Tensor C, cudaStream_t stream)
{
    int M = A.shape(0);
    int K = A.shape(1);
    int N = B.shape(1);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    CHECK_CUDA_ERR(cudaEventRecord(start, stream));
    for (int i = 0; i < NUM_ROUND; i++)
    {
        mllm::kernel::gemm_kernel(&A, &B, &C, stream);
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}
float run_cublas_gemm(Tensor A, Tensor B, Tensor C, cublasHandle_t handle, cudaStream_t stream)
{
    int M = A.shape(0);
    int K = A.shape(1);
    int N = B.shape(1);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUDA_ERR(cudaEventRecord(start, stream));
    for (int i = 0; i < NUM_ROUND; i++)
    {
        CHECK_CUBLAS_ERR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data(), N, A.data(), K, &beta, C.data(), N));
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

    return milliseconds / 1000.0;
}

void warmup_gpu(cudaStream_t stream, cublasHandle_t handle)
{
    const int N = 256;
    Tensor A = Tensor::rand({N, N}, Device::CUDA, false, stream);
    Tensor B = Tensor::rand({N, N}, Device::CUDA, false, stream);
    Tensor C = Tensor::rand({N, N}, Device::CUDA, false, stream);

    for (int i = 0; i < 10; i++)
    {
        run_my_gemm(A, B, C, stream);
        run_cublas_gemm(A, B, C, handle, stream);
    }
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
}

void set_tensor(Tensor t, float val)
{
    t.toDevice(Device::CPU);
    int sz = t.size();
    for (int i = 0; i < sz; i++)
        *t[i] = val;
    t.toDevice(Device::CUDA);
}

void print_mat(Tensor t)
{
    auto ori_device = t.device();
    t.toDevice(Device::CPU);
    for (size_t i = 0; i < t.shape(0) / 16; i++)
    {
        for (size_t j = 0; j < t.shape(1) / 16; j++)
            std::cout << (*t[{i, j}]) << " ";
        std::cout << std::endl;
    }
    t.toDevice(ori_device);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ./gemm_eval eval_type csv_path\n"
                  << "eval_type:\n"
                  << "  0: M=N=K   M % 256 == 0\n"
                  << "  1: M=N=K   M % 4 == 0\n"
                  << "  2: M=N=K   M % 1 == 0\n"
                  << "  3: all of the above\n";
        exit(0);
    }
    int eval_type = atoi(argv[1]);
    std::string csv_path = argv[2];

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    const float EPS = 1e-5;
    const int MAX_X = 6144;
    const int STRIDE = 256;
    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERR(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    std::ofstream csv(csv_path);
    if (eval_type == 0)
        csv << "Size,MyGEMMDiv64,CUBLAS\n";
    else if (eval_type == 1)
        csv << "Size,MyGEMMDiv4,CUBLAS\n";
    else if (eval_type == 2)
        csv << "Size,MyGEMMDiv1,CUBLAS\n";
    else if (eval_type == 3)
        csv << "Size,MyGEMM,CUBLAS\n";

    warmup_gpu(stream, handle);
    float sum_ratio = 0.0f;
    int count = 0;
    for (size_t N = STRIDE; N <= MAX_X; N += STRIDE)
    {
        if (eval_type == 0 || eval_type == 3)
        {
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * NUM_ROUND * 1e-9;

            Tensor A = Tensor::rand({M, K}, Device::CUDA, false, stream);
            Tensor B = Tensor::rand({K, N}, Device::CUDA, false, stream);
            Tensor C1 = Tensor::rand({M, N}, Device::CUDA, false, stream);
            Tensor C2 = Tensor::rand({M, N}, Device::CUDA, false, stream);

            float my_sec = run_my_gemm(A, B, C1, stream);
            float cublas_sec = run_cublas_gemm(A, B, C2, handle, stream);

            float my_gflops = gflo / my_sec;
            float cublas_gflops = gflo / cublas_sec;
            float ratio = cublas_sec / my_sec;
            sum_ratio += ratio;
            count += 1;

            std::cout << "Size: " << N << ", GFLO: " << gflo
                      << ", MyGEMMDiv64: Time=" << my_sec << "sec, Performance=" << my_gflops << " GFLOPS"
                      << ", CUBLAS: Time=" << cublas_sec << "sec, Performance=" << cublas_gflops << " GFLOPS"
                      << ", ratio_cublas=" << ratio
                      << ", avg_ratio=" << (sum_ratio / count)
                      << std::endl;
            csv << N << "," << my_gflops << "," << cublas_gflops << "\n";

            C1.toDevice(Device::CPU);
            C2.toDevice(Device::CPU);
            for (size_t i = 0; i < M * N; i++)
            {
                if (std::fabs(std::fabs(C1.data()[i] / C2.data()[i]) - 1.0f) > EPS)
                {
                    std::cout << "Result mismatch at " << i << ": " << C1.data()[i] << " vs " << C2.data()[i] << "\n";

                    std::cout << "C1:\n";
                    print_mat(C1);
                    std::cout << "C2:\n";
                    print_mat(C2);
                    exit(EXIT_FAILURE);
                }
            }
        }
        if (eval_type == 1 || eval_type == 3)
        {
            N = N + STRIDE - 20;
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * NUM_ROUND * 1e-9;

            Tensor A = Tensor::rand({M, K}, Device::CUDA, false, stream);
            Tensor B = Tensor::rand({K, N}, Device::CUDA, false, stream);
            Tensor C1 = Tensor::rand({M, N}, Device::CUDA, false, stream);
            Tensor C2 = Tensor::rand({M, N}, Device::CUDA, false, stream);

            float my_sec = run_my_gemm(A, B, C1, stream);
            float cublas_sec = run_cublas_gemm(A, B, C2, handle, stream);

            float my_gflops = gflo / my_sec;
            float cublas_gflops = gflo / cublas_sec;
            float ratio = cublas_sec / my_sec;
            sum_ratio += ratio;
            count += 1;

            std::cout << "Size: " << N << ", GFLO: " << gflo
                      << ", MyGEMMDiv4: Time=" << my_sec << "sec, Performance=" << my_gflops << " GFLOPS"
                      << ", CUBLAS: Time=" << cublas_sec << "sec, Performance=" << cublas_gflops << " GFLOPS"
                      << ", ratio_cublas=" << ratio
                      << ", avg_ratio=" << (sum_ratio / count)
                      << std::endl;
            csv << N << "," << my_gflops << "," << cublas_gflops << "\n";

            C1.toDevice(Device::CPU);
            C2.toDevice(Device::CPU);
            for (size_t i = 0; i < M * N; i++)
            {
                if (std::fabs(std::fabs(C1.data()[i] / C2.data()[i]) - 1.0f) > EPS)
                {
                    std::cout << "Result mismatch at " << i << ": " << C1.data()[i] << " vs " << C2.data()[i] << "\n";

                    std::cout << "C1:\n";
                    print_mat(C1);
                    std::cout << "C2:\n";
                    print_mat(C2);
                    exit(EXIT_FAILURE);
                }
            }
            N = N - (STRIDE - 20);
        }
        if (eval_type == 2 || eval_type == 3)
        {
            N = N + STRIDE - 27;
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * NUM_ROUND * 1e-9;

            Tensor A = Tensor::rand({M, K}, Device::CUDA, false, stream);
            Tensor B = Tensor::rand({K, N}, Device::CUDA, false, stream);
            Tensor C1 = Tensor::rand({M, N}, Device::CUDA, false, stream);
            Tensor C2 = Tensor::rand({M, N}, Device::CUDA, false, stream);

            float my_sec = run_my_gemm(A, B, C1, stream);
            float cublas_sec = run_cublas_gemm(A, B, C2, handle, stream);

            float my_gflops = gflo / my_sec;
            float cublas_gflops = gflo / cublas_sec;
            float ratio = cublas_sec / my_sec;
            sum_ratio += ratio;
            count += 1;

            std::cout << "Size: " << N << ", GFLO: " << gflo
                      << ", MyGEMMDiv1: Time=" << my_sec << "sec, Performance=" << my_gflops << " GFLOPS"
                      << ", CUBLAS: Time=" << cublas_sec << "sec, Performance=" << cublas_gflops << " GFLOPS"
                      << ", ratio_cublas=" << ratio
                      << ", avg_ratio=" << (sum_ratio / count)
                      << std::endl;
            csv << N << "," << my_gflops << "," << cublas_gflops << "\n";

            C1.toDevice(Device::CPU);
            C2.toDevice(Device::CPU);
            for (size_t i = 0; i < M * N; i++)
            {
                if (std::fabs(std::fabs(C1.data()[i] / C2.data()[i]) - 1.0f) > EPS)
                {
                    std::cout << "Result mismatch at " << i << ": " << C1.data()[i] << " vs " << C2.data()[i] << "\n";

                    std::cout << "C1:\n";
                    print_mat(C1);
                    std::cout << "C2:\n";
                    print_mat(C2);
                    exit(EXIT_FAILURE);
                }
            }
            N = N - (STRIDE - 27);
        }
    }

    cublasDestroy(handle);
    CHECK_CUDA_ERR(cudaStreamDestroy(stream));
    csv.close();
    google::ShutdownGoogleLogging();

    return 0;
}