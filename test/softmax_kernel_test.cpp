#include "base/tensor.h"
#include "base/util.h"
#include <cmath>
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <cuda_runtime.h>
#include "kernel/kernel.h"
#include "kernel/cuda/softmax_backend_selector.h"
#include "kernel/cuda/softmax_kernel.cuh"
#include "op/softmax.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

namespace
{
    constexpr float kSoftmaxComparisonTolerance = 1e-5f;

    void fill_deterministic_tensor(Tensor &tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            const float value = static_cast<float>((static_cast<int>(i % 29) - 14)) * 0.125f;
            *tensor[i] = value;
        }
    }

    Tensor tensor_from_fp32_as_bf16(const std::vector<float> &values,
                                    const std::vector<size_t> &shape,
                                    Device device,
                                    cudaStream_t stream)
    {
        std::vector<uint16_t> bf16(values.size());
        load_f32_to_bf16(values.data(), bf16.data(), bf16.size());
        return Tensor::from_vector(bf16, shape, device, false, stream);
    }

    std::vector<float> tensor_to_fp32_vector(Tensor tensor)
    {
        tensor.toDevice(Device::CPU);
        std::vector<float> values(tensor.size());
        materialize_float_storage(tensor.raw_data(), tensor.dtype(), values.data(), DType::FP32, values.size());
        return values;
    }

    void expect_tensor_values_near(const Tensor &expected, const Tensor &actual, float tolerance)
    {
        ASSERT_EQ(expected.shape(), actual.shape());
        ASSERT_EQ(expected.size(), actual.size());

        const float *expected_data = const_cast<Tensor &>(expected).data();
        const float *actual_data = const_cast<Tensor &>(actual).data();
        for (size_t i = 0; i < expected.size(); ++i)
        {
            EXPECT_NEAR(expected_data[i], actual_data[i], tolerance) << "at index: " << i;
        }
    }

    void copy_tensor_to_cpu_async(const Tensor &device_tensor, Tensor &host_tensor, cudaStream_t stream)
    {
        ASSERT_EQ(device_tensor.device(), Device::CUDA);
        ASSERT_EQ(host_tensor.device(), Device::CPU);
        ASSERT_EQ(device_tensor.size(), host_tensor.size());

        ASSERT_EQ(cudaMemcpyAsync(host_tensor.data(),
                                  const_cast<Tensor &>(device_tensor).data(),
                                  device_tensor.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  cudaSuccess);
    }

    void fill_large_logit_tensor(Tensor &tensor, float base_value)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            const float offset = static_cast<float>((static_cast<int>(i % 17) - 8)) * 0.25f;
            *tensor[i] = base_value + offset;
        }
    }

    kernel::SoftmaxBackendExecution expected_public_softmax_execution()
    {
        if (kernel::cuda_softmax_library_available())
        {
            return kernel::SoftmaxBackendExecution::LibraryBacked;
        }
#if defined(MLLM_CUDNN_AVAILABLE) && MLLM_CUDNN_AVAILABLE
        return kernel::SoftmaxBackendExecution::LibraryBacked;
#else
        return kernel::SoftmaxBackendExecution::HandwrittenFallback;
#endif
    }

    void compute_stable_softmax_reference(const Tensor &input, Tensor &output)
    {
        ASSERT_EQ(input.device(), Device::CPU);
        ASSERT_EQ(output.device(), Device::CPU);
        ASSERT_EQ(input.shape(), output.shape());

        const size_t num_mats = const_cast<Tensor &>(input).num_mats();
        const size_t rows = input.shape(-2);
        const size_t cols = input.shape(-1);

        for (size_t mat = 0; mat < num_mats; ++mat)
        {
            const float *input_mat = const_cast<Tensor &>(input).mat(mat);
            float *output_mat = output.mat(mat);
            for (size_t row = 0; row < rows; ++row)
            {
                const float *input_row = input_mat + row * cols;
                float *output_row = output_mat + row * cols;

                float row_max = input_row[0];
                for (size_t col = 1; col < cols; ++col)
                {
                    row_max = std::max(row_max, input_row[col]);
                }

                float row_sum = 0.0f;
                for (size_t col = 0; col < cols; ++col)
                {
                    const float shifted = std::exp(input_row[col] - row_max);
                    output_row[col] = shifted;
                    row_sum += shifted;
                }

                for (size_t col = 0; col < cols; ++col)
                {
                    output_row[col] /= row_sum;
                }
            }
        }
    }

    void expect_rows_sum_to_one_and_finite(const Tensor &tensor, float tolerance)
    {
        ASSERT_EQ(tensor.device(), Device::CPU);

        const size_t num_mats = const_cast<Tensor &>(tensor).num_mats();
        const size_t rows = tensor.shape(-2);
        const size_t cols = tensor.shape(-1);

        for (size_t mat = 0; mat < num_mats; ++mat)
        {
            const float *tensor_mat = const_cast<Tensor &>(tensor).mat(mat);
            for (size_t row = 0; row < rows; ++row)
            {
                const float *tensor_row = tensor_mat + row * cols;
                float row_sum = 0.0f;
                for (size_t col = 0; col < cols; ++col)
                {
                    EXPECT_TRUE(std::isfinite(tensor_row[col])) << "mat=" << mat << " row=" << row << " col=" << col;
                    row_sum += tensor_row[col];
                }
                EXPECT_NEAR(row_sum, 1.0f, tolerance) << "mat=" << mat << " row=" << row;
            }
        }
    }

    void run_public_softmax_parity_case(const std::vector<size_t> &shape,
                                        cudaStream_t stream,
                                        float base_value)
    {
        Tensor handwritten_input(shape, Device::CPU, false, stream);
        fill_large_logit_tensor(handwritten_input, base_value);

        Tensor public_input = handwritten_input.clone();
        Tensor handwritten_output(shape, Device::CUDA, false, stream);
        Tensor public_output(shape, Device::CUDA, false, stream);
        Tensor handwritten_host(shape, Device::CPU, false, stream);
        Tensor public_host(shape, Device::CPU, false, stream);

        handwritten_input.toDevice(Device::CUDA);
        public_input.toDevice(Device::CUDA);

        kernel::reset_last_softmax_backend_execution();
        kernel::softmax_kernel_cuda_handwritten(&handwritten_input, &handwritten_output, stream);
        copy_tensor_to_cpu_async(handwritten_output, handwritten_host, stream);

        kernel::reset_last_softmax_backend_execution();
        kernel::get_softmax_kernel(Device::CUDA)(&public_input, &public_output, stream);
        copy_tensor_to_cpu_async(public_output, public_host, stream);

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        expect_tensor_values_near(handwritten_host, public_host, kSoftmaxComparisonTolerance);
        EXPECT_EQ(kernel::get_last_softmax_backend_execution(), expected_public_softmax_execution());
        EXPECT_TRUE(kernel::saw_softmax_backend_execution(expected_public_softmax_execution()));
    }
}

class SoftmaxTest : public ::testing::Test
{
protected:
    vector<size_t> shape;
    vector<float> data;
    Tensor a;
    op::Softmax softmax_cpu;
    op::Softmax softmax_cuda;

    SoftmaxTest() : shape({2, 2, 4}),
                    data({1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0,

                          1.0, 6.0, 7.0, 4.0,
                          5.0, 2.0, 3.0, 8.0}),
                    a(data.data(), shape, true, Device::CPU, false, nullptr),
                    softmax_cpu(Device::CPU, nullptr),
                    softmax_cuda(Device::CUDA, nullptr)
    {
        google::InitGoogleLogging("SoftmaxTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up SoftmaxTest";
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }

    void pt_ts3(Tensor &ts)
    {
        for (size_t i = 0; i < ts.size(); i++)
        {
            cout << *ts[i] << ' ';
            if ((i + 1) % ts.shape(-1) == 0)
                cout << endl;
            if ((i + 1) % (ts.shape(-1) * ts.shape(-2)) == 0)
                cout << endl;
        }
        cout << endl;
    }
};

TEST_F(SoftmaxTest, CPU)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running softmax kernel on CPU...";
    softmax_cpu.forward(a, a);

    cout << "softmax(a):\n";
    pt_ts3(a);
}

TEST_F(SoftmaxTest, CUDA)
{
    cout << "a:\n";
    pt_ts3(a);

    VLOG(DEBUG) << "Running softmax kernel on CUDA...";
    a.toDevice(Device::CUDA);
    softmax_cuda.forward(a, a);
    a.toDevice(Device::CPU);

    cout << "softmax(a):\n";
    pt_ts3(a);
}

class SoftmaxCheck : public ::testing::Test
{
protected:
    vector<size_t> shape;
    Tensor a;
    float check_eps = 1e-6;
    op::Softmax softmax_cpu;
    op::Softmax softmax_cuda;

    SoftmaxCheck() : shape({8, 8, 1024}),
                     a(shape, Device::CPU, false, nullptr),
                     softmax_cpu(Device::CPU, nullptr),
                     softmax_cuda(Device::CUDA, nullptr)
    {
        google::InitGoogleLogging("SoftmaxTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Set up SoftmaxTest";

        size_t total_size = a.size();
        for (size_t i = 0; i < total_size; i++)
            *a[i] = base::get_random_float();
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }

    void pt_ts3(Tensor &ts)
    {
        for (size_t i = 0; i < ts.size(); i++)
        {
            cout << *ts[i] << ' ';
            if ((i + 1) % ts.shape(-1) == 0)
                cout << endl;
            if ((i + 1) % (ts.shape(-1) * ts.shape(-2)) == 0)
                cout << endl;
        }
        cout << endl;
    }
};

TEST_F(SoftmaxCheck, CPUvsCUDA)
{
    auto b = a.clone();
    b.toDevice(Device::CUDA);

    VLOG(DEBUG) << "Running softmax kernel on CPU...";
    softmax_cpu.forward(a, a);
    VLOG(DEBUG) << "Running softmax kernel on CUDA...";
    softmax_cuda.forward(b, b);
    b.toDevice(Device::CPU);

    int diff = 0;
    int size = a.size();
    for (int i = 0; i < size; i++)
    {
        diff += std::fabs(*a[i] - *b[i]) > check_eps;
    }
    EXPECT_EQ(diff, 0);
}

TEST(SoftmaxKernelAccessorTest, CudaAccessorDoesNotReturnHandwrittenKernelDirectly)
{
    EXPECT_NE(kernel::get_softmax_kernel(Device::CUDA), kernel::softmax_kernel_cuda_handwritten);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxMatchesHandwrittenOnExplicitCudaStream)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    Tensor handwritten_input({2, 3, 11}, Device::CPU, false, stream);
    fill_deterministic_tensor(handwritten_input);

    Tensor public_input = handwritten_input.clone();
    Tensor handwritten_output({2, 3, 11}, Device::CUDA, false, stream);
    Tensor public_output({2, 3, 11}, Device::CUDA, false, stream);
    Tensor handwritten_host({2, 3, 11}, Device::CPU, false, stream);
    Tensor public_host({2, 3, 11}, Device::CPU, false, stream);

    handwritten_input.toDevice(Device::CUDA);
    public_input.toDevice(Device::CUDA);

    kernel::reset_last_softmax_backend_execution();
    kernel::softmax_kernel_cuda_handwritten(&handwritten_input, &handwritten_output, stream);
    copy_tensor_to_cpu_async(handwritten_output, handwritten_host, stream);

    kernel::reset_last_softmax_backend_execution();
    kernel::get_softmax_kernel(Device::CUDA)(&public_input, &public_output, stream);
    copy_tensor_to_cpu_async(public_output, public_host, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    expect_tensor_values_near(handwritten_host, public_host, kSoftmaxComparisonTolerance);
    EXPECT_EQ(kernel::get_last_softmax_backend_execution(), expected_public_softmax_execution());
    EXPECT_TRUE(kernel::saw_softmax_backend_execution(expected_public_softmax_execution()));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxMatchesHandwrittenForLargeLogits)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const std::vector<std::vector<size_t>> shapes = {
        {32, 1, 128},
        {1, 151936},
    };

    for (const auto &shape : shapes)
    {
        SCOPED_TRACE(::testing::Message() << "shape rank=" << shape.size() << " size=" << shape.back());
        run_public_softmax_parity_case(shape, stream, 80.0f);
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxRemainsStableForVeryLargeLogits)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    const std::vector<std::vector<size_t>> shapes = {
        {32, 1, 128},
        {1, 151936},
    };

    for (const auto &shape : shapes)
    {
        SCOPED_TRACE(::testing::Message() << "shape rank=" << shape.size() << " size=" << shape.back());

        Tensor host_input(shape, Device::CPU, false, stream);
        Tensor reference_output(shape, Device::CPU, false, stream);
        Tensor public_input(shape, Device::CPU, false, stream);
        Tensor public_output(shape, Device::CUDA, false, stream);
        Tensor public_host(shape, Device::CPU, false, stream);

        fill_large_logit_tensor(host_input, 120.0f);
        public_input = host_input.clone();
        compute_stable_softmax_reference(host_input, reference_output);

        public_input.toDevice(Device::CUDA);

        kernel::reset_last_softmax_backend_execution();
        kernel::get_softmax_kernel(Device::CUDA)(&public_input, &public_output, stream);
        copy_tensor_to_cpu_async(public_output, public_host, stream);

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        expect_rows_sum_to_one_and_finite(public_host, 5e-4f);
        expect_tensor_values_near(reference_output, public_host, 1e-4f);
        EXPECT_EQ(kernel::get_last_softmax_backend_execution(), expected_public_softmax_execution());
    }

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxMaterializesTransposedTensorBeforeFallback)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor public_input({7, 11}, Device::CPU, false, stream);
    fill_deterministic_tensor(public_input);
    public_input.t();

    Tensor public_output({7, 11}, Device::CPU, false, stream);
    public_output.t();

    Tensor reference_input = public_input.clone();
    Tensor reference_output(reference_input.shape(), Device::CUDA, false, stream);
    Tensor reference_host(reference_input.shape(), Device::CPU, false, stream);
    Tensor public_host(public_input.shape(), Device::CPU, false, stream);

    reference_input.contiguous();
    reference_input.toDevice(Device::CUDA);
    public_input.toDevice(Device::CUDA);
    public_output.toDevice(Device::CUDA);

    kernel::softmax_kernel_cuda_handwritten(&reference_input, &reference_output, stream);
    copy_tensor_to_cpu_async(reference_output, reference_host, stream);

    kernel::reset_last_softmax_backend_execution();
    kernel::get_softmax_kernel(Device::CUDA)(&public_input, &public_output, stream);
    copy_tensor_to_cpu_async(public_output, public_host, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    EXPECT_TRUE(public_input.is_contiguous());
    EXPECT_TRUE(public_output.is_contiguous());
    expect_tensor_values_near(reference_host, public_host, kSoftmaxComparisonTolerance);
    EXPECT_EQ(kernel::get_last_softmax_backend_execution(), kernel::SoftmaxBackendExecution::HandwrittenFallback);
    EXPECT_TRUE(kernel::saw_softmax_backend_execution(kernel::SoftmaxBackendExecution::HandwrittenFallback));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxSupportsInPlaceExecution)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor reference_input({2, 3, 11}, Device::CPU, false, stream);
    fill_deterministic_tensor(reference_input);

    Tensor inplace_tensor = reference_input.clone();
    Tensor reference_output({2, 3, 11}, Device::CUDA, false, stream);
    Tensor reference_host({2, 3, 11}, Device::CPU, false, stream);
    Tensor inplace_host({2, 3, 11}, Device::CPU, false, stream);

    reference_input.toDevice(Device::CUDA);
    inplace_tensor.toDevice(Device::CUDA);

    kernel::softmax_kernel_cuda_handwritten(&reference_input, &reference_output, stream);
    copy_tensor_to_cpu_async(reference_output, reference_host, stream);

    kernel::reset_last_softmax_backend_execution();
    kernel::get_softmax_kernel(Device::CUDA)(&inplace_tensor, &inplace_tensor, stream);
    copy_tensor_to_cpu_async(inplace_tensor, inplace_host, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    expect_tensor_values_near(reference_host, inplace_host, kSoftmaxComparisonTolerance);
    EXPECT_EQ(kernel::get_last_softmax_backend_execution(), expected_public_softmax_execution());
    EXPECT_TRUE(kernel::saw_softmax_backend_execution(expected_public_softmax_execution()));

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(SoftmaxKernelExecutionTraceTest, PublicCudaSoftmaxRecordsFallbackForIneligibleShape)
{
    Tensor input({2, 3, 4, 5}, Device::CPU, false, nullptr);
    fill_deterministic_tensor(input);
    Tensor output({2, 3, 4, 5}, Device::CPU, false, nullptr);

    input.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);

    kernel::reset_last_softmax_backend_execution();
    kernel::get_softmax_kernel(Device::CUDA)(&input, &output, nullptr);

    EXPECT_EQ(kernel::get_last_softmax_backend_execution(), kernel::SoftmaxBackendExecution::HandwrittenFallback);
    EXPECT_TRUE(kernel::saw_softmax_backend_execution(kernel::SoftmaxBackendExecution::HandwrittenFallback));
}

TEST(SoftmaxBackendSelectionTest, SelectsLibraryBackendForEligibleRank2CudaTensor)
{
    Tensor input({7, 11}, Device::CUDA, false, nullptr);
    Tensor output({7, 11}, Device::CUDA, false, nullptr);

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::LibraryBacked);
    EXPECT_TRUE(decision.reason.empty());
}

TEST(SoftmaxBackendSelectionTest, SelectsLibraryBackendForEligibleRank3CudaTensor)
{
    Tensor input({2, 5, 11}, Device::CUDA, false, nullptr);
    Tensor output({2, 5, 11}, Device::CUDA, false, nullptr);

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::LibraryBacked);
    EXPECT_TRUE(decision.reason.empty());
}

TEST(SoftmaxBackendSelectionTest, FallsBackForCpuTensor)
{
    Tensor input({7, 11}, Device::CPU, false, nullptr);
    Tensor output({7, 11}, Device::CPU, false, nullptr);

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::HandwrittenFallback);
    EXPECT_FALSE(decision.uses_library_backend());
    EXPECT_NE(decision.reason.find("CUDA"), std::string::npos);
}

TEST(SoftmaxBackendSelectionTest, FallsBackForTransposedCudaTensor)
{
    Tensor input({7, 11}, Device::CUDA, false, nullptr);
    Tensor output({11, 7}, Device::CUDA, false, nullptr);

    input.t();
    output.t();

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::HandwrittenFallback);
    EXPECT_NE(decision.reason.find("contiguous"), std::string::npos);
}

TEST(SoftmaxBackendSelectionTest, FallsBackForRankOutsideRollout)
{
    Tensor input({2, 3, 4, 5}, Device::CUDA, false, nullptr);
    Tensor output({2, 3, 4, 5}, Device::CUDA, false, nullptr);

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::HandwrittenFallback);
    EXPECT_NE(decision.reason.find("rank 2 and rank 3"), std::string::npos);
}

TEST(SoftmaxBackendSelectionTest, FallsBackWhenLibraryIsUnavailable)
{
    Tensor input({7, 11}, Device::CUDA, false, nullptr);
    Tensor output({7, 11}, Device::CUDA, false, nullptr);

    kernel::SoftmaxBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = false;

    const auto decision = kernel::select_softmax_backend(input, output, options);

    EXPECT_EQ(decision.backend, kernel::SoftmaxBackend::HandwrittenFallback);
    EXPECT_NE(decision.reason.find("unavailable"), std::string::npos);
}

TEST(SoftmaxKernelBF16Test, PublicKernelSupportsBF16StorageOnCpuAndCuda)
{
    const std::vector<size_t> shape = {2, 4};
    const std::vector<float> values = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 0.0f, 1.0f, 2.0f,
    };

    Tensor input_fp32(shape, Device::CPU, false, nullptr);
    Tensor expected(shape, Device::CPU, false, nullptr);
    std::copy(values.begin(), values.end(), input_fp32.data());
    kernel::get_softmax_kernel(Device::CPU)(&input_fp32, &expected, nullptr);

    Tensor cpu_input = tensor_from_fp32_as_bf16(values, shape, Device::CPU, nullptr);
    Tensor cpu_output(shape, Device::CPU, false, nullptr, DType::BF16);
    kernel::get_softmax_kernel(Device::CPU)(&cpu_input, &cpu_output, nullptr);

    EXPECT_EQ(cpu_output.dtype(), DType::BF16);
    const std::vector<float> cpu_values = tensor_to_fp32_vector(cpu_output);
    const float *expected_values = expected.data();
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(cpu_values[i], expected_values[i], 2e-2f) << "cpu index " << i;
    }

    Tensor cuda_input = cpu_input.clone();
    Tensor cuda_output(shape, Device::CUDA, false, nullptr, DType::BF16);
    cuda_input.toDevice(Device::CUDA);
    kernel::reset_last_softmax_backend_execution();
    kernel::get_softmax_kernel(Device::CUDA)(&cuda_input, &cuda_output, nullptr);

    EXPECT_EQ(cuda_output.dtype(), DType::BF16);
    const std::vector<float> cuda_values = tensor_to_fp32_vector(cuda_output);
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(cuda_values[i], expected_values[i], 2e-2f) << "cuda index " << i;
    }
    EXPECT_NE(kernel::get_last_softmax_backend_execution(), kernel::SoftmaxBackendExecution::Unknown);
}
