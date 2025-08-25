#include "kernel/kernel.h"
#include "base/tensor.h"
#include "op/add.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <armadillo>

using namespace mllm;
using namespace mllm::base;

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

class MatAddTest : public ::testing::Test
{
protected:
    std::vector<size_t> large_shape;
    std::vector<size_t> small_shape;

    void SetUp() override
    {
        google::InitGoogleLogging("MatAddTest");
        FLAGS_logtostderr = true;
        small_shape = {4, 8, 4};
        large_shape = {2, 4, 2048, 1024};
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(MatAddTest, CPUAddSmall)
{
    Tensor input0(small_shape);
    Tensor input1(small_shape);
    Tensor output(small_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    // 执行加法操作
    op::Add add_op(Device::CPU, nullptr);
    add_op.forward(input0, input1, output);

    VLOG(TRACE) << "CPU ADD Small Result:";
    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        VLOG(TRACE) << "Index: " << i << ", Input0: " << *input0[i] << ", Input1: " << *input1[i] << ", Output: " << *output[i];
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}
TEST_F(MatAddTest, CPUTransposeAddSmall)
{
    Tensor input0(small_shape);
    auto tshape = small_shape;
    std::swap(tshape[1], tshape[2]);
    Tensor input1(tshape);
    Tensor output(small_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }
    input0.t();
    output.t();

    // 执行加法操作
    op::Add add_op(Device::CPU, nullptr);
    add_op.forward(input0, input1, output);

    VLOG(TRACE) << "CPU ADD Small Result:";
    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        VLOG(TRACE) << "Index: " << i << ", Input0: " << *input0[i] << ", Input1: " << *input1[i] << ", Output: " << *output[i];
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}

TEST_F(MatAddTest, CUDAAddSmall)
{
    Tensor input0(small_shape);
    Tensor input1(small_shape);
    Tensor output(small_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    input0.toDevice(Device::CUDA);
    input1.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);
    // 执行加法操作
    op::Add add_op(Device::CUDA, nullptr);
    add_op.forward(input0, input1, output);
    input0.toDevice(Device::CPU);
    input1.toDevice(Device::CPU);
    output.toDevice(Device::CPU);
    VLOG(TRACE) << "CUDA ADD Small Result:";
    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        VLOG(TRACE) << "Index: " << i << ", Input0: " << *input0[i] << ", Input1: " << *input1[i] << ", Output: " << *output[i];
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}
TEST_F(MatAddTest, CUDATransposeAddSmall)
{
    Tensor input0(small_shape);
    auto tshape = small_shape;
    std::swap(tshape[1], tshape[2]);
    Tensor input1(tshape);
    Tensor output(small_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    input0.toDevice(Device::CUDA);
    input1.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);
    input0.t();
    output.t();
    // 执行加法操作
    op::Add add_op(Device::CUDA, nullptr);
    add_op.forward(input0, input1, output);
    input0.toDevice(Device::CPU);
    input1.toDevice(Device::CPU);
    output.toDevice(Device::CPU);

    VLOG(TRACE) << "CUDA ADD Small Result:";
    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        VLOG(TRACE) << "Index: " << i << ", Input0: " << *input0[i] << ", Input1: " << *input1[i] << ", Output: " << *output[i];
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}

TEST_F(MatAddTest, CUDAAddLarge)
{
    Tensor input0(large_shape);
    Tensor input1(large_shape);
    Tensor output(large_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    input0.toDevice(Device::CUDA);
    input1.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);
    // 执行加法操作
    op::Add add_op(Device::CUDA, nullptr);
    add_op.forward(input0, input1, output);
    input0.toDevice(Device::CPU);
    input1.toDevice(Device::CPU);
    output.toDevice(Device::CPU);

    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}

TEST_F(MatAddTest, CUDATransposeAddLarge)
{
    Tensor input0(large_shape);
    auto tshape = large_shape;
    std::swap(tshape[2], tshape[3]);
    Tensor input1(tshape);
    Tensor output(large_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    input0.toDevice(Device::CUDA);
    input1.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);
    input0.t();
    output.t();
    // 执行加法操作
    op::Add add_op(Device::CUDA, nullptr);
    add_op.forward(input0, input1, output);
    input0.toDevice(Device::CPU);
    input1.toDevice(Device::CPU);
    output.toDevice(Device::CPU);

    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}

TEST_F(MatAddTest, CPUAddLarge)
{
    Tensor input0(large_shape);
    Tensor input1(large_shape);
    Tensor output(large_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }

    // 执行加法操作
    op::Add add_op(Device::CPU, nullptr);
    add_op.forward(input0, input1, output);

    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}

TEST_F(MatAddTest, CPUTransposeAddLarge)
{
    Tensor input0(large_shape);
    auto tshape = large_shape;
    std::swap(tshape[2], tshape[3]);
    Tensor input1(tshape);
    Tensor output(large_shape);

    // 初始化输入数据
    size_t total_size = input0.size();
    for (size_t i = 0; i < total_size; ++i)
    {
        *input0[i] = static_cast<float>(i % 100);
        *input1[i] = static_cast<float>((i * 2) % 100);
    }
    input0.t();
    output.t();

    // 执行加法操作
    op::Add add_op(Device::CPU, nullptr);
    add_op.forward(input0, input1, output);

    // 验证输出结果
    for (size_t i = 0; i < total_size; ++i)
    {
        float expected = *input0[i] + *input1[i];
        EXPECT_FLOAT_EQ(*output[i], expected) << "at index " << i;
    }
}