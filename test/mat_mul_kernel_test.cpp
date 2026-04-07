#include "kernel/kernel.h"
#include "kernel/cuda/mat_mul_kernel.cuh"
#include "kernel/cuda/mat_mul_backend_selector.h"
#include "base/util.h"
#include "base/tensor.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace mllm;
using namespace mllm::base;

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace
{
    constexpr float kMatMulComparisonTolerance = 1e-3f;

    struct MatMulParityCase
    {
        std::vector<size_t> lhs_shape;
        std::vector<size_t> rhs_shape;
    };

    void fill_deterministic_tensor(Tensor &tensor)
    {
        for (size_t i = 0; i < tensor.size(); ++i)
        {
            const float value = static_cast<float>((static_cast<int>(i % 23) - 11)) * 0.03125f;
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

    void run_cuda_matmul_parity_case(const std::vector<size_t> &lhs_shape, const std::vector<size_t> &rhs_shape, cudaStream_t stream)
    {
        ASSERT_EQ(lhs_shape.size(), 2u);
        ASSERT_EQ(rhs_shape.size(), 2u);
        ASSERT_EQ(lhs_shape[1], rhs_shape[0]);

        const std::vector<size_t> out_shape = {lhs_shape[0], rhs_shape[1]};

        Tensor lhs_cpu(lhs_shape, Device::CPU, false, stream);
        Tensor rhs_cpu(rhs_shape, Device::CPU, false, stream);
        fill_deterministic_tensor(lhs_cpu);
        fill_deterministic_tensor(rhs_cpu);

        Tensor lhs_cuda = lhs_cpu.clone();
        Tensor rhs_cuda = rhs_cpu.clone();
        lhs_cuda.toDevice(Device::CUDA);
        rhs_cuda.toDevice(Device::CUDA);

        Tensor handwritten_output(out_shape, Device::CUDA, false, stream);
        Tensor library_output(out_shape, Device::CUDA, false, stream);

        kernel::mat_mul_kernel_cuda_vec(&lhs_cuda, &rhs_cuda, &handwritten_output, stream);
        kernel::mat_mul_kernel_cuda_cublas(&lhs_cuda, &rhs_cuda, &library_output, stream);

        if (stream)
        {
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        }
        else
        {
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        }

        handwritten_output.toDevice(Device::CPU);
        library_output.toDevice(Device::CPU);

        expect_tensor_values_near(handwritten_output, library_output, kMatMulComparisonTolerance);
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
}

class SmallMatMulTest : public ::testing::Test
{
protected:
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    std::vector<size_t> shape_o;
    Tensor cpu_input0;
    Tensor cpu_input1;
    Tensor cpu_output;
    Tensor cuda_input0;
    Tensor cuda_input1;
    Tensor cuda_output;

    std::mt19937 rnd;
    std::uniform_real_distribution<> urd;
    SmallMatMulTest() : rnd(std::random_device{}()), urd(0.0, 1.0) {}

    void SetUp() override
    {
        google::InitGoogleLogging("SmallMatMulTest");

        FLAGS_logtostderr = true;
        // shape_a = {16, 8};
        // shape_b = {8, 8};
        // shape_o = {16, 8};
        // shape_a = {4, 4};
        // shape_b = {4, 2};
        // shape_o = {4, 2};
        shape_a = {128, 128};
        shape_b = {128, 32};
        shape_o = {128, 32};
        cpu_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cpu_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cpu_output = Tensor(shape_o, Device::CPU, false, nullptr);
        cuda_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cuda_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cuda_output = Tensor(shape_o, Device::CPU, false, nullptr);

        size_t size0 = cpu_input0.size();
        size_t size1 = cpu_input1.size();
        for (size_t i = 0; i < size0; i++)
        {
            *cuda_input0[i] = *cpu_input0[i] = urd(rnd);
        }
        for (size_t i = 0; i < size1; i++)
        {
            *cuda_input1[i] = *cpu_input1[i] = urd(rnd);
        }
        cuda_input0.toDevice(Device::CUDA);
        cuda_input1.toDevice(Device::CUDA);
        cuda_output.toDevice(Device::CUDA);
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};
class LargeMatMulTest : public ::testing::Test
{
protected:
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    std::vector<size_t> shape_o;
    Tensor cpu_input0;
    Tensor cpu_input1;
    Tensor cpu_output;
    Tensor cuda_input0;
    Tensor cuda_input1;
    Tensor cuda_output;

    std::mt19937 rnd;
    std::uniform_real_distribution<> urd;
    LargeMatMulTest() : rnd(std::random_device{}()), urd(0.0, 1.0) {}

    void SetUp() override
    {
        google::InitGoogleLogging("MatMulTest");

        FLAGS_logtostderr = true;
        shape_a = {4096, 1024};
        shape_b = {1024, 4096};
        shape_o = {4096, 4096};

        cpu_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cpu_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cpu_output = Tensor(shape_o, Device::CPU, false, nullptr);
        cuda_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cuda_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cuda_output = Tensor(shape_o, Device::CPU, false, nullptr);
        size_t size0 = cpu_input0.size();
        size_t size1 = cpu_input1.size();
        for (size_t i = 0; i < size0; i++)
        {
            *cuda_input0[i] = *cpu_input0[i] = urd(rnd);
        }
        for (size_t i = 0; i < size1; i++)
        {
            *cuda_input1[i] = *cpu_input1[i] = urd(rnd);
        }

        cuda_input0.toDevice(Device::CUDA);
        cuda_input1.toDevice(Device::CUDA);
        cuda_output.toDevice(Device::CUDA);
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};
class SmallTensorMulTest : public ::testing::Test
{
protected:
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    std::vector<size_t> shape_o;
    Tensor cpu_input0;
    Tensor cpu_input1;
    Tensor cpu_output;
    Tensor cuda_input0;
    Tensor cuda_input1;
    Tensor cuda_output;

    std::mt19937 rnd;
    std::uniform_real_distribution<> urd;
    SmallTensorMulTest() : rnd(std::random_device{}()), urd(0.0, 1.0) {}

    void SetUp() override
    {
        google::InitGoogleLogging("SmallMatMulTest");

        FLAGS_logtostderr = true;
        // shape_a = {16, 8};
        // shape_b = {8, 8};
        // shape_o = {16, 8};
        // shape_a = {4, 4};
        // shape_b = {4, 2};
        // shape_o = {4, 2};
        shape_a = {4, 2, 128, 128};
        shape_b = {4, 2, 128, 32};
        shape_o = {4, 2, 128, 32};
        cpu_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cpu_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cpu_output = Tensor(shape_o, Device::CPU, false, nullptr);
        cuda_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cuda_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cuda_output = Tensor(shape_o, Device::CPU, false, nullptr);

        size_t size0 = cpu_input0.size();
        size_t size1 = cpu_input1.size();
        for (size_t i = 0; i < size0; i++)
        {
            *cuda_input0[i] = *cpu_input0[i] = urd(rnd);
        }
        for (size_t i = 0; i < size1; i++)
        {
            *cuda_input1[i] = *cpu_input1[i] = urd(rnd);
        }
        cuda_input0.toDevice(Device::CUDA);
        cuda_input1.toDevice(Device::CUDA);
        cuda_output.toDevice(Device::CUDA);
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};
class LargeTensorMulTest : public ::testing::Test
{
protected:
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    std::vector<size_t> shape_o;
    Tensor cpu_input0;
    Tensor cpu_input1;
    Tensor cpu_output;
    Tensor cuda_input0;
    Tensor cuda_input1;
    Tensor cuda_output;

    std::mt19937 rnd;
    std::uniform_real_distribution<> urd;
    LargeTensorMulTest() : rnd(std::random_device{}()), urd(0.0, 1.0) {}

    void SetUp() override
    {
        google::InitGoogleLogging("MatMulTest");

        FLAGS_logtostderr = true;
        shape_a = {4, 2, 4096, 1024};
        shape_b = {4, 2, 1024, 4096};
        shape_o = {4, 2, 4096, 4096};

        cpu_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cpu_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cpu_output = Tensor(shape_o, Device::CPU, false, nullptr);
        cuda_input0 = Tensor(shape_a, Device::CPU, false, nullptr);
        cuda_input1 = Tensor(shape_b, Device::CPU, false, nullptr);
        cuda_output = Tensor(shape_o, Device::CPU, false, nullptr);
        size_t size0 = cpu_input0.size();
        size_t size1 = cpu_input1.size();
        for (size_t i = 0; i < size0; i++)
        {
            *cuda_input0[i] = *cpu_input0[i] = urd(rnd);
        }
        for (size_t i = 0; i < size1; i++)
        {
            *cuda_input1[i] = *cpu_input1[i] = urd(rnd);
        }

        cuda_input0.toDevice(Device::CUDA);
        cuda_input1.toDevice(Device::CUDA);
        cuda_output.toDevice(Device::CUDA);
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};

TEST(CudaLibraryMatMulAdapterTest, Dense2dMatchesCpuMatMul)
{
    Tensor cpu_input0({3, 4}, Device::CPU, false, nullptr);
    Tensor cpu_input1({4, 2}, Device::CPU, false, nullptr);
    Tensor cpu_output({3, 2}, Device::CPU, false, nullptr);
    Tensor cuda_input0({3, 4}, Device::CPU, false, nullptr);
    Tensor cuda_input1({4, 2}, Device::CPU, false, nullptr);
    Tensor cuda_output({3, 2}, Device::CPU, false, nullptr);

    const std::vector<float> lhs = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    const std::vector<float> rhs = {
        1.0f, 0.0f,
        0.5f, 1.0f,
        1.5f, -1.0f,
        2.0f, 0.25f,
    };

    for (size_t i = 0; i < lhs.size(); ++i)
    {
        *cpu_input0[i] = lhs[i];
        *cuda_input0[i] = lhs[i];
    }
    for (size_t i = 0; i < rhs.size(); ++i)
    {
        *cpu_input1[i] = rhs[i];
        *cuda_input1[i] = rhs[i];
    }

    cuda_input0.toDevice(Device::CUDA);
    cuda_input1.toDevice(Device::CUDA);
    cuda_output.toDevice(Device::CUDA);

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::mat_mul_kernel_cuda_cublas(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    for (size_t i = 0; i < cpu_output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
    }
}

TEST(CudaLibraryMatMulAdapterTest, Dense2dBackendsMatchAcrossRepresentativeShapes)
{
    const std::vector<MatMulParityCase> cases = {
        {{3, 7}, {7, 5}},
        {{8, 15}, {15, 4}},
        {{17, 31}, {31, 11}},
    };

    for (const auto &matmul_case : cases)
    {
        SCOPED_TRACE(::testing::Message() << "lhs=" << matmul_case.lhs_shape[0] << "x" << matmul_case.lhs_shape[1]
                                          << ", rhs=" << matmul_case.rhs_shape[0] << "x" << matmul_case.rhs_shape[1]);
        run_cuda_matmul_parity_case(matmul_case.lhs_shape, matmul_case.rhs_shape, nullptr);
    }
}

TEST(CudaLibraryMatMulAdapterTest, Dense2dPublicEntrypointMatchesHandwrittenOnExplicitCudaStream)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    Tensor lhs_cpu({11, 13}, Device::CPU, false, stream);
    Tensor rhs_cpu({13, 9}, Device::CPU, false, stream);
    fill_deterministic_tensor(lhs_cpu);
    fill_deterministic_tensor(rhs_cpu);

    Tensor lhs_cuda = lhs_cpu.clone();
    Tensor rhs_cuda = rhs_cpu.clone();
    lhs_cuda.toDevice(Device::CUDA);
    rhs_cuda.toDevice(Device::CUDA);

    Tensor handwritten_output({11, 9}, Device::CUDA, false, stream);
    Tensor public_output({11, 9}, Device::CUDA, false, stream);
    Tensor handwritten_host({11, 9}, Device::CPU, false, stream);
    Tensor public_host({11, 9}, Device::CPU, false, stream);

    kernel::reset_last_mat_mul_backend_execution();
    kernel::mat_mul_kernel_cuda_vec(&lhs_cuda, &rhs_cuda, &handwritten_output, stream);
    copy_tensor_to_cpu_async(handwritten_output, handwritten_host, stream);

    kernel::reset_last_mat_mul_backend_execution();
    kernel::get_mat_mul_kernel(Device::CUDA)(&lhs_cuda, &rhs_cuda, &public_output, stream);
    copy_tensor_to_cpu_async(public_output, public_host, stream);

    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    expect_tensor_values_near(handwritten_host, public_host, kMatMulComparisonTolerance);
    EXPECT_EQ(kernel::get_last_mat_mul_backend_execution(), kernel::MatMulBackendExecution::LibraryBacked);

    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(MatMulKernelAccessorTest, CudaAccessorDoesNotReturnHandwrittenKernelDirectly)
{
    EXPECT_NE(kernel::get_mat_mul_kernel(Device::CUDA), kernel::mat_mul_kernel_cuda_vec);
}

TEST(MatMulKernelExecutionTraceTest, RecordsExecutedBackendPathForLibraryAndFallbackCalls)
{
    Tensor library_input0({3, 4}, Device::CPU, false, nullptr);
    Tensor library_input1({4, 2}, Device::CPU, false, nullptr);
    Tensor library_output({3, 2}, Device::CPU, false, nullptr);

    Tensor fallback_input0({2, 3, 4}, Device::CPU, false, nullptr);
    Tensor fallback_input1({2, 4, 2}, Device::CPU, false, nullptr);
    Tensor fallback_output({2, 3, 2}, Device::CPU, false, nullptr);

    for (size_t i = 0; i < library_input0.size(); ++i)
    {
        *library_input0[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < library_input1.size(); ++i)
    {
        *library_input1[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < fallback_input0.size(); ++i)
    {
        *fallback_input0[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < fallback_input1.size(); ++i)
    {
        *fallback_input1[i] = static_cast<float>(i + 1);
    }

    library_input0.toDevice(Device::CUDA);
    library_input1.toDevice(Device::CUDA);
    library_output.toDevice(Device::CUDA);
    fallback_input0.toDevice(Device::CUDA);
    fallback_input1.toDevice(Device::CUDA);
    fallback_output.toDevice(Device::CUDA);

    kernel::reset_last_mat_mul_backend_execution();
    kernel::get_mat_mul_kernel(Device::CUDA)(&library_input0, &library_input1, &library_output, nullptr);
    EXPECT_EQ(kernel::get_last_mat_mul_backend_execution(), kernel::MatMulBackendExecution::LibraryBacked);
    EXPECT_TRUE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::LibraryBacked));
    EXPECT_FALSE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::HandwrittenFallback));

    kernel::get_mat_mul_kernel(Device::CUDA)(&fallback_input0, &fallback_input1, &fallback_output, nullptr);
    EXPECT_EQ(kernel::get_last_mat_mul_backend_execution(), kernel::MatMulBackendExecution::HandwrittenFallback);
    EXPECT_TRUE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::LibraryBacked));
    EXPECT_TRUE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::HandwrittenFallback));

    kernel::reset_last_mat_mul_backend_execution();
    kernel::get_mat_mul_kernel(Device::CUDA)(&fallback_input0, &fallback_input1, &fallback_output, nullptr);
    EXPECT_EQ(kernel::get_last_mat_mul_backend_execution(), kernel::MatMulBackendExecution::HandwrittenFallback);
    EXPECT_TRUE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::HandwrittenFallback));
    EXPECT_FALSE(kernel::saw_mat_mul_backend_execution(kernel::MatMulBackendExecution::LibraryBacked));
}

TEST_F(SmallMatMulTest, SmallMatMulCheck)
{
    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}
TEST_F(SmallMatMulTest, SmallTransposeMatMulCheck)
{
    cpu_input0.t();
    cpu_input1.t();
    cpu_output.t();
    cuda_input0.t();
    cuda_input1.t();
    cuda_output.t();

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input1, &cpu_input0, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input1, &cuda_input0, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}

TEST_F(LargeMatMulTest, LargeMatMulCheck)
{
    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}
TEST_F(LargeMatMulTest, LargeTransposeMatMulCheck)
{
    cpu_input0.t();
    cpu_input1.t();
    cpu_output.t();
    cuda_input0.t();
    cuda_input1.t();
    cuda_output.t();

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input1, &cpu_input0, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input1, &cuda_input0, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}
TEST_F(SmallTensorMulTest, SmallMatMulCheck)
{
    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}
TEST_F(SmallTensorMulTest, SmallTransposeMatMulCheck)
{
    cpu_input0.t();
    cpu_input1.t();
    cpu_output.t();
    cuda_input0.t();
    cuda_input1.t();
    cuda_output.t();

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input1, &cpu_input0, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input1, &cuda_input0, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}

TEST_F(LargeTensorMulTest, LargeMatMulCheck)
{
    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}
TEST_F(LargeTensorMulTest, LargeTransposeMatMulCheck)
{
    cpu_input0.t();
    cpu_input1.t();
    cpu_output.t();
    cuda_input0.t();
    cuda_input1.t();
    cuda_output.t();

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input1, &cpu_input0, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input1, &cuda_input0, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_NEAR(*cpu_output[i], *cuda_output[i], kMatMulComparisonTolerance) << "at index: " << i;
    }
}

TEST(MatMulBackendSelectionTest, SelectsLibraryBackendForEligibleCudaDenseMatMul)
{
    Tensor lhs({2, 3}, Device::CUDA, false, nullptr);
    Tensor rhs({3, 4}, Device::CUDA, false, nullptr);
    Tensor out({2, 4}, Device::CUDA, false, nullptr);

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::LibraryBacked);
    EXPECT_TRUE(decision.reason.empty());
}

TEST(MatMulBackendSelectionTest, FallsBackForCpuTensorsWithReason)
{
    Tensor lhs({2, 3}, Device::CPU, false, nullptr);
    Tensor rhs({3, 4}, Device::CPU, false, nullptr);
    Tensor out({2, 4}, Device::CPU, false, nullptr);

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::HandwrittenFallback);
    EXPECT_FALSE(decision.uses_library_backend());
    EXPECT_FALSE(decision.reason.empty());
}

TEST(MatMulBackendSelectionTest, FallsBackForTransposedCudaTensors)
{
    Tensor lhs({2, 3}, Device::CUDA, false, nullptr);
    Tensor rhs({3, 4}, Device::CUDA, false, nullptr);
    Tensor out({3, 2}, Device::CUDA, false, nullptr);

    lhs.t();
    rhs.t();
    out.t();

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::HandwrittenFallback);
    EXPECT_FALSE(decision.reason.empty());
    EXPECT_NE(decision.reason.find("contiguous"), std::string::npos);
}

TEST(MatMulBackendSelectionTest, FallsBackForBatchedShapeMismatch)
{
    Tensor lhs({2, 3, 4}, Device::CUDA, false, nullptr);
    Tensor rhs({3, 4, 5}, Device::CUDA, false, nullptr);
    Tensor out({2, 3, 5}, Device::CUDA, false, nullptr);

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;
    options.allow_batched_matmul = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::HandwrittenFallback);
    EXPECT_FALSE(decision.reason.empty());
    EXPECT_NE(decision.reason.find("2D-only"), std::string::npos);
}

TEST(MatMulBackendSelectionTest, FallsBackForAnyBatchedCudaTensors)
{
    Tensor lhs({2, 3, 4}, Device::CUDA, false, nullptr);
    Tensor rhs({2, 4, 5}, Device::CUDA, false, nullptr);
    Tensor out({2, 3, 5}, Device::CUDA, false, nullptr);

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;
    options.allow_batched_matmul = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::HandwrittenFallback);
    EXPECT_FALSE(decision.reason.empty());
    EXPECT_NE(decision.reason.find("2D-only"), std::string::npos);
}

TEST(MatMulBackendSelectionTest, FallsBackForScalarRhs)
{
    Tensor lhs({2, 3}, Device::CUDA, false, nullptr);
    Tensor rhs({1}, Device::CUDA, false, nullptr);
    Tensor out({2, 3}, Device::CUDA, false, nullptr);

    kernel::MatMulBackendSelectionOptions options;
    options.library_enabled = true;
    options.library_available = true;

    const auto decision = kernel::select_mat_mul_backend(lhs, rhs, out, options);

    EXPECT_EQ(decision.backend, kernel::MatMulBackend::HandwrittenFallback);
    EXPECT_NE(decision.reason.find("scalar rhs"), std::string::npos);
}

TEST(MatMulKernelBF16Test, PublicKernelSupportsBF16StorageOnCpuAndCuda)
{
    const std::vector<size_t> lhs_shape = {2, 3};
    const std::vector<size_t> rhs_shape = {3, 2};
    const std::vector<size_t> out_shape = {2, 2};
    const std::vector<float> lhs_values = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const std::vector<float> rhs_values = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f,
    };
    const std::vector<float> expected = {
        58.0f, 64.0f,
        139.0f, 154.0f,
    };

    Tensor cpu_lhs = tensor_from_fp32_as_bf16(lhs_values, lhs_shape, Device::CPU, nullptr);
    Tensor cpu_rhs = tensor_from_fp32_as_bf16(rhs_values, rhs_shape, Device::CPU, nullptr);
    Tensor cpu_out(out_shape, Device::CPU, false, nullptr, DType::BF16);

    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_lhs, &cpu_rhs, &cpu_out, nullptr);

    EXPECT_EQ(cpu_out.dtype(), DType::BF16);
    const std::vector<float> cpu_values = tensor_to_fp32_vector(cpu_out);
    ASSERT_EQ(cpu_values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(cpu_values[i], expected[i], kMatMulComparisonTolerance) << "cpu index " << i;
    }

    Tensor cuda_lhs = cpu_lhs.clone();
    Tensor cuda_rhs = cpu_rhs.clone();
    Tensor cuda_out(out_shape, Device::CUDA, false, nullptr, DType::BF16);
    cuda_lhs.toDevice(Device::CUDA);
    cuda_rhs.toDevice(Device::CUDA);

    kernel::reset_last_mat_mul_backend_execution();
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_lhs, &cuda_rhs, &cuda_out, nullptr);

    EXPECT_EQ(cuda_out.dtype(), DType::BF16);
    const std::vector<float> cuda_values = tensor_to_fp32_vector(cuda_out);
    ASSERT_EQ(cuda_values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(cuda_values[i], expected[i], kMatMulComparisonTolerance) << "cuda index " << i;
    }

    EXPECT_NE(kernel::get_last_mat_mul_backend_execution(), kernel::MatMulBackendExecution::Unknown);
}
