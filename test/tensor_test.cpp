#include "base/tensor.h"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>

using namespace mllm::base;

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code if needed
    }

    void TearDown() override
    {
        // Cleanup code if needed
    }
};

// Test Tensor construction with CPU device
TEST_F(TensorTest, ConstructorCPU)
{
    std::vector<size_t> shape = {2, 3, 4};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 24); // 2 * 3 * 4 = 24
    EXPECT_NE(tensor.data(), nullptr);
}

// Test Tensor construction with CUDA device
TEST_F(TensorTest, ConstructorCUDA)
{
    std::vector<size_t> shape = {5, 5};
    Tensor tensor(shape, Device::CUDA);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 25); // 5 * 5 = 25
    EXPECT_NE(tensor.data(), nullptr);
}

// Test Tensor construction with mutable flag
TEST_F(TensorTest, ConstructorMutable)
{
    std::vector<size_t> shape = {3, 3};
    Tensor tensor(shape, Device::CPU, true);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 9); // 3 * 3 = 9
    EXPECT_NE(tensor.data(), nullptr);
}

// Test Tensor construction with 1D shape
TEST_F(TensorTest, Constructor1D)
{
    std::vector<size_t> shape = {10};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 10);
    EXPECT_NE(tensor.data(), nullptr);
}

// Test Tensor construction with 4D shape
TEST_F(TensorTest, Constructor4D)
{
    std::vector<size_t> shape = {2, 3, 4, 5};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 120); // 2 * 3 * 4 * 5 = 120
    EXPECT_NE(tensor.data(), nullptr);
}

// Test Tensor with zero dimension
TEST_F(TensorTest, ConstructorZeroDimension)
{
    std::vector<size_t> shape = {0};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 0);
}

// Test Tensor with single element
TEST_F(TensorTest, ConstructorSingleElement)
{
    std::vector<size_t> shape = {1, 1, 1};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 1);
    EXPECT_NE(tensor.data(), nullptr);
}

// Test shape method
TEST_F(TensorTest, ShapeMethod)
{
    std::vector<size_t> shape = {7, 8, 9};
    Tensor tensor(shape, Device::CPU);

    const auto &returned_shape = tensor.shape();
    EXPECT_EQ(returned_shape.size(), 3);
    EXPECT_EQ(returned_shape[0], 7);
    EXPECT_EQ(returned_shape[1], 8);
    EXPECT_EQ(returned_shape[2], 9);
}

// Test size calculation
TEST_F(TensorTest, SizeCalculation)
{
    // Test different shapes
    std::vector<std::vector<size_t>> shapes = {
        {1},         // size = 1
        {5},         // size = 5
        {2, 3},      // size = 6
        {2, 3, 4},   // size = 24
        {1, 10, 1},  // size = 10
        {2, 2, 2, 2} // size = 16
    };

    std::vector<size_t> expected_sizes = {1, 5, 6, 24, 10, 16};

    for (size_t i = 0; i < shapes.size(); ++i)
    {
        Tensor tensor(shapes[i], Device::CPU);
        EXPECT_EQ(tensor.size(), expected_sizes[i])
            << "Failed for shape index " << i;
    }
}

// Test data access and modification
TEST_F(TensorTest, DataAccess)
{
    std::vector<size_t> shape = {2, 2};
    Tensor tensor(shape, Device::CPU);

    float *data = tensor.data();
    ASSERT_NE(data, nullptr);

    // Write and read data
    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
    EXPECT_FLOAT_EQ(data[2], 3.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);
}

// Test large tensor
TEST_F(TensorTest, LargeTensor)
{
    std::vector<size_t> shape = {100, 100};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 10000);
    EXPECT_NE(tensor.data(), nullptr);

    // Test data access for large tensor
    float *data = tensor.data();
    data[0] = 42.0f;
    data[9999] = 24.0f;

    EXPECT_FLOAT_EQ(data[0], 42.0f);
    EXPECT_FLOAT_EQ(data[9999], 24.0f);
}

// Test tensor copy semantics (if applicable)
TEST_F(TensorTest, MultipleTensors)
{
    std::vector<size_t> shape1 = {3, 3};
    std::vector<size_t> shape2 = {2, 5};

    Tensor tensor1(shape1, Device::CPU);
    Tensor tensor2(shape2, Device::CPU);

    EXPECT_EQ(tensor1.size(), 9);
    EXPECT_EQ(tensor2.size(), 10);

    // Ensure they have different data pointers
    EXPECT_NE(tensor1.data(), tensor2.data());
}

// Test edge case: empty shape vector (if supported)
TEST_F(TensorTest, EmptyShape)
{
    std::vector<size_t> shape = {};

    // This might throw an exception or create a scalar tensor
    // depending on implementation. Adjust test accordingly.
    try
    {
        Tensor tensor(shape, Device::CPU);
        // If construction succeeds, check reasonable behavior
        EXPECT_EQ(tensor.shape(), shape);
    }
    catch (const std::exception &e)
    {
        // If construction throws, that's also acceptable behavior
        SUCCEED() << "Empty shape properly rejected: " << e.what();
    }
}
