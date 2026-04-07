#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "base/tensor.h"
#include "kernel/kernel.h"
#include "kernel/cuda/gemm_kernel.cuh"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/mat_mul_backend_selector.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace mllm;
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

namespace
{
    constexpr int kSyntheticRounds = 500;
    constexpr float kParityAbsTolerance = 1e-3f;
    constexpr float kParityRelTolerance = 1e-3f;
    constexpr double kBenchmarkTargetFlops = 5.0e8;
    constexpr double kWarmupTargetFlops = 5.0e7;
    constexpr int kMaxBenchmarkRounds = 10;
    constexpr int kMaxWarmupRounds = 3;

    struct InferenceWorkload
    {
        std::string workload;
        size_t seq_len;
        size_t m;
        size_t k;
        size_t n;
        int warmup_rounds;
        int benchmark_rounds;
    };

    struct TimedRun
    {
        float seconds = 0.0f;
        float gflops = 0.0f;
        float avg_ms = 0.0f;
    };

    struct ParityStats
    {
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
    };

    void fill_deterministic_tensor(Tensor &tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            const float value = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.03125f;
            *tensor[i] = value;
        }
    }

    void print_mat(Tensor t)
    {
        auto ori_device = t.device();
        t.toDevice(Device::CPU);
        for (size_t i = 0; i < t.shape(0) / 16; i++)
        {
            for (size_t j = 0; j < t.shape(1) / 16; j++)
            {
                std::cout << (*t[{i, j}]) << " ";
            }
            std::cout << std::endl;
        }
        t.toDevice(ori_device);
    }

    int clamp_rounds(int rounds, int max_rounds)
    {
        return std::max(1, std::min(rounds, max_rounds));
    }

    int compute_rounds(size_t m, size_t k, size_t n, double target_flops, int max_rounds)
    {
        const double flops_per_call = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n);
        if (flops_per_call <= 0.0)
        {
            return 1;
        }
        const int rounds = static_cast<int>(std::floor(target_flops / flops_per_call));
        return clamp_rounds(rounds, max_rounds);
    }

    float compute_gflops(size_t m, size_t k, size_t n, int rounds, float seconds)
    {
        if (seconds <= 0.0f)
        {
            return 0.0f;
        }
        const double total_flops = 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n) * static_cast<double>(rounds);
        return static_cast<float>(total_flops * 1e-9 / static_cast<double>(seconds));
    }

    const char *backend_execution_to_string(kernel::MatMulBackendExecution execution)
    {
        switch (execution)
        {
        case kernel::MatMulBackendExecution::HandwrittenFallback:
            return "HandwrittenFallback";
        case kernel::MatMulBackendExecution::LibraryBacked:
            return "LibraryBacked";
        case kernel::MatMulBackendExecution::Unknown:
        default:
            return "Unknown";
        }
    }

    void sync_stream(cudaStream_t stream)
    {
        if (stream)
        {
            CHECK_CUDA_ERR(cudaStreamSynchronize(stream));
        }
        else
        {
            CHECK_CUDA_ERR(cudaDeviceSynchronize());
        }
    }

    float run_my_gemm(Tensor A, Tensor B, Tensor C, cudaStream_t stream)
    {
        int M = A.shape(0);
        int K = A.shape(1);
        int N = B.shape(1);

        cudaEvent_t start, stop;
        CHECK_CUDA_ERR(cudaEventCreate(&start));
        CHECK_CUDA_ERR(cudaEventCreate(&stop));

        CHECK_CUDA_ERR(cudaEventRecord(start, stream));
        for (int i = 0; i < kSyntheticRounds; i++)
        {
            mllm::kernel::gemm_kernel(&A, &B, &C, stream);
        }
        CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERR(cudaGetLastError());

        float milliseconds = 0;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

        CHECK_CUDA_ERR(cudaEventDestroy(start));
        CHECK_CUDA_ERR(cudaEventDestroy(stop));

        return milliseconds / 1000.0f;
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
        for (int i = 0; i < kSyntheticRounds; i++)
        {
            CHECK_CUBLAS_ERR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B.data(), N, A.data(), K, &beta, C.data(), N));
        }
        CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERR(cudaGetLastError());

        float milliseconds = 0;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));

        CHECK_CUDA_ERR(cudaEventDestroy(start));
        CHECK_CUDA_ERR(cudaEventDestroy(stop));

        return milliseconds / 1000.0f;
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

    void verify_synthetic_outputs_match(Tensor &handwritten, Tensor &cublas)
    {
        handwritten.toDevice(Device::CPU);
        cublas.toDevice(Device::CPU);
        for (size_t i = 0; i < handwritten.size(); i++)
        {
            if (std::fabs(std::fabs(handwritten.data()[i] / cublas.data()[i]) - 1.0f) > kParityAbsTolerance)
            {
                std::cout << "Result mismatch at " << i << ": " << handwritten.data()[i] << " vs " << cublas.data()[i] << "\n";
                std::cout << "C1:\n";
                print_mat(handwritten);
                std::cout << "C2:\n";
                print_mat(cublas);
                exit(EXIT_FAILURE);
            }
        }
    }

    TimedRun run_timed_matmul(kernel::MatMulKernel kernel_fn,
                              Tensor &lhs,
                              Tensor &rhs,
                              Tensor &output,
                              cudaStream_t stream,
                              int rounds,
                              bool expect_library_backend)
    {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        CHECK_CUDA_ERR(cudaEventCreate(&start));
        CHECK_CUDA_ERR(cudaEventCreate(&stop));

        if (expect_library_backend)
        {
            kernel::reset_last_mat_mul_backend_execution();
        }

        CHECK_CUDA_ERR(cudaEventRecord(start, stream));
        for (int i = 0; i < rounds; ++i)
        {
            kernel_fn(&lhs, &rhs, &output, stream);
        }
        CHECK_CUDA_ERR(cudaEventRecord(stop, stream));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
        CHECK_CUDA_ERR(cudaGetLastError());

        if (expect_library_backend)
        {
            if (!kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::LibraryBacked))
            {
                std::cerr << "Expected public CUDA matmul window to include a library-backed execution." << std::endl;
                exit(EXIT_FAILURE);
            }
            if (kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::HandwrittenFallback))
            {
                std::cerr << "Public CUDA matmul benchmark window unexpectedly used handwritten fallback." << std::endl;
                exit(EXIT_FAILURE);
            }

            const auto execution = kernel::get_last_mat_mul_backend_execution();
            if (execution != kernel::MatMulBackendExecution::LibraryBacked)
            {
                std::cerr << "Expected public CUDA matmul to execute the library-backed path, got "
                          << backend_execution_to_string(execution) << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        float milliseconds = 0.0f;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
        CHECK_CUDA_ERR(cudaEventDestroy(start));
        CHECK_CUDA_ERR(cudaEventDestroy(stop));

        TimedRun run;
        run.seconds = milliseconds / 1000.0f;
        run.avg_ms = milliseconds / static_cast<float>(rounds);
        run.gflops = compute_gflops(lhs.shape(0), lhs.shape(1), rhs.shape(1), rounds, run.seconds);
        return run;
    }

    void warmup_inference_case(Tensor &lhs,
                               Tensor &rhs,
                               Tensor &handwritten_output,
                               Tensor &library_output,
                               cudaStream_t stream,
                               int rounds)
    {
        for (int i = 0; i < rounds; ++i)
        {
            kernel::mat_mul_kernel_cuda_vec(&lhs, &rhs, &handwritten_output, stream);
            kernel::reset_last_mat_mul_backend_execution();
            kernel::get_mat_mul_kernel(Device::CUDA)(&lhs, &rhs, &library_output, stream);
            if (!kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::LibraryBacked) ||
                kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::HandwrittenFallback))
            {
                std::cerr << "Warmup expected a pure library-backed execution window." << std::endl;
                exit(EXIT_FAILURE);
            }

            const auto execution = kernel::get_last_mat_mul_backend_execution();
            if (execution != kernel::MatMulBackendExecution::LibraryBacked)
            {
                std::cerr << "Warmup expected library-backed execution but got "
                          << backend_execution_to_string(execution) << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        sync_stream(stream);
    }

    ParityStats verify_inference_outputs_match(Tensor &handwritten_output, Tensor &library_output)
    {
        handwritten_output.toDevice(Device::CPU);
        library_output.toDevice(Device::CPU);

        ParityStats stats;
        for (size_t i = 0; i < handwritten_output.size(); ++i)
        {
            const float handwritten_value = handwritten_output.data()[i];
            const float library_value = library_output.data()[i];
            const float abs_diff = std::fabs(handwritten_value - library_value);
            const float denom = std::max(std::fabs(handwritten_value), 1.0e-6f);
            const float rel_diff = abs_diff / denom;

            stats.max_abs_diff = std::max(stats.max_abs_diff, abs_diff);
            stats.max_rel_diff = std::max(stats.max_rel_diff, rel_diff);

            if (abs_diff > kParityAbsTolerance && rel_diff > kParityRelTolerance)
            {
                std::cerr << "Inference parity mismatch at index " << i
                          << ": handwritten=" << handwritten_value
                          << ", library_first=" << library_value
                          << ", abs_diff=" << abs_diff
                          << ", rel_diff=" << rel_diff << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        return stats;
    }

    std::vector<InferenceWorkload> build_qwen3_inference_workloads(const std::string &filter)
    {
        constexpr size_t kHiddenSize = 1024;
        constexpr size_t kIntermediateSize = 3072;
        constexpr size_t kVocabSize = 151936;

        struct ProjectionShape
        {
            const char *workload;
            size_t k;
            size_t n;
            std::vector<size_t> seq_lens;
        };

        const std::vector<ProjectionShape> shapes = {
            {"q_proj", kHiddenSize, kHiddenSize, {1, 8, 32}},
            {"v_proj", kHiddenSize, kHiddenSize, {1, 8, 32}},
            {"o_proj", kHiddenSize, kHiddenSize, {1, 8, 32}},
            {"up_proj", kHiddenSize, kIntermediateSize, {1, 8, 32}},
            {"gate_proj", kHiddenSize, kIntermediateSize, {1, 8, 32}},
            {"down_proj", kIntermediateSize, kHiddenSize, {1, 8, 32}},
            {"lm_head", kHiddenSize, kVocabSize, {1, 8}},
        };

        std::vector<InferenceWorkload> workloads;
        for (const auto &shape : shapes)
        {
            for (size_t seq_len : shape.seq_lens)
            {
                const std::string case_name = std::string(shape.workload) + "_seq" + std::to_string(seq_len);
                if (!filter.empty() &&
                    case_name.find(filter) == std::string::npos &&
                    std::string(shape.workload).find(filter) == std::string::npos)
                {
                    continue;
                }

                InferenceWorkload workload;
                workload.workload = shape.workload;
                workload.seq_len = seq_len;
                workload.m = seq_len;
                workload.k = shape.k;
                workload.n = shape.n;
                workload.warmup_rounds = compute_rounds(workload.m, workload.k, workload.n, kWarmupTargetFlops, kMaxWarmupRounds);
                workload.benchmark_rounds = compute_rounds(workload.m, workload.k, workload.n, kBenchmarkTargetFlops, kMaxBenchmarkRounds);
                workloads.push_back(workload);
            }
        }

        return workloads;
    }

    void run_qwen3_inference_benchmark(const std::string &csv_path, const std::string &filter)
    {
        auto workloads = build_qwen3_inference_workloads(filter);
        if (workloads.empty())
        {
            std::cerr << "No Qwen3 inference workloads matched filter '" << filter << "'." << std::endl;
            exit(EXIT_FAILURE);
        }

        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERR(cudaStreamCreate(&stream));

        std::ofstream csv(csv_path);
        csv << "suite,workload,seq_len,m,k,n,rounds,handwritten_avg_ms,handwritten_gflops,library_first_avg_ms,library_first_gflops,speedup,backend_execution,max_abs_diff,max_rel_diff\n";

        std::cout << std::fixed << std::setprecision(4);

        for (const auto &workload : workloads)
        {
            Tensor lhs({workload.m, workload.k}, Device::CPU, false, stream);
            Tensor rhs({workload.k, workload.n}, Device::CPU, false, stream);
            fill_deterministic_tensor(lhs);
            fill_deterministic_tensor(rhs);
            lhs.toDevice(Device::CUDA);
            rhs.toDevice(Device::CUDA);

            Tensor handwritten_output({workload.m, workload.n}, Device::CUDA, false, stream);
            Tensor library_output({workload.m, workload.n}, Device::CUDA, false, stream);

            warmup_inference_case(lhs, rhs, handwritten_output, library_output, stream, workload.warmup_rounds);

            const auto handwritten_run = run_timed_matmul(kernel::mat_mul_kernel_cuda_vec,
                                                          lhs,
                                                          rhs,
                                                          handwritten_output,
                                                          stream,
                                                          workload.benchmark_rounds,
                                                          false);
            const auto library_run = run_timed_matmul(kernel::get_mat_mul_kernel(Device::CUDA),
                                                      lhs,
                                                      rhs,
                                                      library_output,
                                                      stream,
                                                      workload.benchmark_rounds,
                                                      true);

            sync_stream(stream);
            const auto parity = verify_inference_outputs_match(handwritten_output, library_output);
            const auto execution = kernel::get_last_mat_mul_backend_execution();
            const float speedup = library_run.seconds > 0.0f ? handwritten_run.seconds / library_run.seconds : 0.0f;

            std::cout << "[qwen3_inference] workload=" << workload.workload
                      << ", seq_len=" << workload.seq_len
                      << ", shape=" << workload.m << "x" << workload.k << " * " << workload.k << "x" << workload.n
                      << ", rounds=" << workload.benchmark_rounds
                      << ", handwritten_avg_ms=" << handwritten_run.avg_ms
                      << ", library_first_avg_ms=" << library_run.avg_ms
                      << ", speedup=" << speedup
                      << ", backend=" << backend_execution_to_string(execution)
                      << ", max_abs_diff=" << parity.max_abs_diff
                      << ", max_rel_diff=" << parity.max_rel_diff
                      << std::endl;

            csv << "qwen3_inference,"
                << workload.workload << ","
                << workload.seq_len << ","
                << workload.m << ","
                << workload.k << ","
                << workload.n << ","
                << workload.benchmark_rounds << ","
                << handwritten_run.avg_ms << ","
                << handwritten_run.gflops << ","
                << library_run.avg_ms << ","
                << library_run.gflops << ","
                << speedup << ","
                << backend_execution_to_string(execution) << ","
                << parity.max_abs_diff << ","
                << parity.max_rel_diff << "\n";
        }

        csv.close();
        CHECK_CUDA_ERR(cudaStreamDestroy(stream));
    }

    void print_usage()
    {
        std::cout << "Usage: ./gemm_eval eval_type csv_path [workload_filter]\n"
                  << "eval_type:\n"
                  << "  0: M=N=K   M % 256 == 0\n"
                  << "  1: M=N=K   M % 4 == 0\n"
                  << "  2: M=N=K   M % 1 == 0\n"
                  << "  3: all of the above\n"
                  << "  4: qwen3-like inference workloads via public CUDA matmul dispatch\n"
                  << "  qwen3_inference: alias for mode 4\n"
                  << "workload_filter:\n"
                  << "  optional substring filter for inference workloads, e.g. q_proj or seq1\n";
    }
} // namespace

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4)
    {
        print_usage();
        return 0;
    }

    const std::string eval_type_arg = argv[1];
    const std::string csv_path = argv[2];
    const std::string workload_filter = argc == 4 ? argv[3] : "";

    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    int eval_type = -1;
    if (eval_type_arg == "qwen3_inference")
    {
        eval_type = 4;
    }
    else
    {
        eval_type = std::atoi(eval_type_arg.c_str());
    }

    if (eval_type == 4)
    {
        run_qwen3_inference_benchmark(csv_path, workload_filter);
        google::ShutdownGoogleLogging();
        return 0;
    }

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
    else
    {
        print_usage();
        cublasDestroy(handle);
        CHECK_CUDA_ERR(cudaStreamDestroy(stream));
        csv.close();
        google::ShutdownGoogleLogging();
        return 1;
    }

    warmup_gpu(stream, handle);
    float sum_ratio = 0.0f;
    int count = 0;
    for (size_t N = STRIDE; N <= MAX_X; N += STRIDE)
    {
        if (eval_type == 0 || eval_type == 3)
        {
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * kSyntheticRounds * 1e-9f;

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

            verify_synthetic_outputs_match(C1, C2);
        }
        if (eval_type == 1 || eval_type == 3)
        {
            N = N + STRIDE - 20;
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * kSyntheticRounds * 1e-9f;

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

            verify_synthetic_outputs_match(C1, C2);
            N = N - (STRIDE - 20);
        }
        if (eval_type == 2 || eval_type == 3)
        {
            N = N + STRIDE - 27;
            const size_t M = N, K = N;
            float gflo = 2.0 * M * N * K * kSyntheticRounds * 1e-9f;

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

            verify_synthetic_outputs_match(C1, C2);
            N = N - (STRIDE - 27);
        }
    }

    cublasDestroy(handle);
    CHECK_CUDA_ERR(cudaStreamDestroy(stream));
    csv.close();
    google::ShutdownGoogleLogging();

    return 0;
}
