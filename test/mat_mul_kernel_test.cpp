#include "kernel/kernel.h"
#include "base/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace mllm;
using namespace mllm::base;

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

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
        cpu_input0 = Tensor(shape_a);
        cpu_input1 = Tensor(shape_b);
        cpu_output = Tensor(shape_o);
        cuda_input0 = Tensor(shape_a);
        cuda_input1 = Tensor(shape_b);
        cuda_output = Tensor(shape_o);

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

        cpu_input0 = Tensor(shape_a);
        cpu_input1 = Tensor(shape_b);
        cpu_output = Tensor(shape_o);
        cuda_input0 = Tensor(shape_a);
        cuda_input1 = Tensor(shape_b);
        cuda_output = Tensor(shape_o);
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
        cpu_input0 = Tensor(shape_a);
        cpu_input1 = Tensor(shape_b);
        cpu_output = Tensor(shape_o);
        cuda_input0 = Tensor(shape_a);
        cuda_input1 = Tensor(shape_b);
        cuda_output = Tensor(shape_o);

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

        cpu_input0 = Tensor(shape_a);
        cpu_input1 = Tensor(shape_b);
        cpu_output = Tensor(shape_o);
        cuda_input0 = Tensor(shape_a);
        cuda_input1 = Tensor(shape_b);
        cuda_output = Tensor(shape_o);
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

TEST_F(SmallMatMulTest, SmallMatMulCheck)
{
    kernel::get_mat_mul_kernel(Device::CPU)(&cpu_input0, &cpu_input1, &cpu_output, nullptr);
    kernel::get_mat_mul_kernel(Device::CUDA)(&cuda_input0, &cuda_input1, &cuda_output, nullptr);

    cuda_output.toDevice(Device::CPU);
    size_t total_size = cpu_output.size();
    for (size_t i = 0; i < total_size; i++)
    {
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
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
        EXPECT_FLOAT_EQ(*cpu_output[i], *cuda_output[i]) << "at index: " << i;
        break;
    }
}