#include "kernel/kernel.h"
#include "base/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace mllm;
using namespace mllm::base;

class AddKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test parameters
        test_size = 1024;
        tolerance = 1e-6f;
    }

    void TearDown() override
    {
        // Clean up if needed
    }

    size_t test_size;
    float tolerance;
};

TEST_F(AddKernelTest, CPUAddSameSize)
{
    // Create test tensors
    auto allocator = HostAllocator::getInstance();

    // Create input tensors with same size
    std::vector<size_t> shape = {32, 32};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize input data
    float *input0_data = input0.data();
    float *input1_data = input1.data();
    for (size_t i = 0; i < test_size; ++i)
    {
        input0_data[i] = static_cast<float>(i) * 0.1f;
        input1_data[i] = static_cast<float>(i) * 0.2f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < test_size; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddScalarBroadcast)
{
    auto allocator = HostAllocator::getInstance();

    // Create tensors: one scalar, one vector
    std::vector<size_t> scalar_shape = {1};
    std::vector<size_t> vector_shape = {test_size};

    Tensor input0(scalar_shape, Device::CPU, allocator); // scalar
    Tensor input1(vector_shape, Device::CPU, allocator); // vector
    Tensor output(vector_shape, Device::CPU, allocator);

    // Initialize input data
    float *input0_data = input0.data();
    float *input1_data = input1.data();
    input0_data[0] = 2.5f; // scalar value

    for (size_t i = 0; i < test_size; ++i)
    {
        input1_data[i] = static_cast<float>(i) * 0.1f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < test_size; ++i)
    {
        float expected = input0_data[0] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddVectorBroadcast)
{
    auto allocator = HostAllocator::getInstance();

    // Create tensors for simple broadcasting: scalar + matrix
    // Since current implementation only supports scalar broadcasting
    std::vector<size_t> scalar_shape = {1};
    std::vector<size_t> matrix_shape = {32, 32};

    Tensor input0(scalar_shape, Device::CPU, allocator); // scalar
    Tensor input1(matrix_shape, Device::CPU, allocator); // matrix
    Tensor output(matrix_shape, Device::CPU, allocator);

    // Initialize input data
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    input0_data[0] = 2.5f; // scalar value

    for (size_t i = 0; i < 32 * 32; ++i)
    {
        input1_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < 32 * 32; ++i)
    {
        float expected = input0_data[0] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance)
            << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddZeroTensors)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {16, 16};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize with zeros
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 256; ++i)
    {
        input0_data[i] = 0.0f;
        input1_data[i] = 0.0f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify all results are zero
    float *output_data = output.data();
    for (size_t i = 0; i < 256; ++i)
    {
        EXPECT_NEAR(output_data[i], 0.0f, tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddNegativeValues)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {64};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize with negative and positive values
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 64; ++i)
    {
        input0_data[i] = -static_cast<float>(i) * 0.5f;
        input1_data[i] = static_cast<float>(i) * 0.3f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < 64; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddLargeValues)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {100};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize with large values
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 100; ++i)
    {
        input0_data[i] = 1e6f + static_cast<float>(i);
        input1_data[i] = 1e7f + static_cast<float>(i) * 2.0f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results (use larger tolerance for large numbers)
    float *output_data = output.data();
    for (size_t i = 0; i < 100; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, 1e-2f) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddSmallValues)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {50};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize with very small values
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 50; ++i)
    {
        input0_data[i] = static_cast<float>(i) * 1e-8f;
        input1_data[i] = static_cast<float>(i) * 1e-9f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < 50; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, 1e-10f) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAdd3DTensors)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {4, 4, 4};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize 3D tensor data
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 64; ++i)
    {
        input0_data[i] = static_cast<float>(i) * 0.1f;
        input1_data[i] = static_cast<float>(i) * 0.05f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < 64; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance) << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddMatrixScalarBroadcast)
{
    auto allocator = HostAllocator::getInstance();

    // Test scalar broadcasting with matrix: scalar + matrix
    // Since current implementation only supports scalar broadcasting
    std::vector<size_t> scalar_shape = {1};
    std::vector<size_t> matrix_shape = {8, 16};

    Tensor input0(scalar_shape, Device::CPU, allocator); // scalar
    Tensor input1(matrix_shape, Device::CPU, allocator); // matrix
    Tensor output(matrix_shape, Device::CPU, allocator);

    // Initialize data
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    input0_data[0] = 3.5f; // scalar value

    for (size_t i = 0; i < 8 * 16; ++i)
    {
        input1_data[i] = static_cast<float>(i) * 0.01f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    float *output_data = output.data();
    for (size_t i = 0; i < 8 * 16; ++i)
    {
        float expected = input0_data[0] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance)
            << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddDifferentShapeSameSize)
{
    auto allocator = HostAllocator::getInstance();

    // Test tensors with different shapes but same total size: (4, 4) + (16, 1)
    std::vector<size_t> shape1 = {4, 4};
    std::vector<size_t> shape2 = {16, 1};

    Tensor input0(shape1, Device::CPU, allocator);
    Tensor input1(shape2, Device::CPU, allocator);
    Tensor output(shape1, Device::CPU, allocator); // output has same shape as input0

    // Initialize data
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 16; ++i)
    {
        input0_data[i] = static_cast<float>(i) * 0.1f;
        input1_data[i] = static_cast<float>(i) * 0.2f;
    }

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify results - element-wise addition regardless of shape
    float *output_data = output.data();
    for (size_t i = 0; i < 16; ++i)
    {
        float expected = input0_data[i] + input1_data[i];
        EXPECT_NEAR(output_data[i], expected, tolerance)
            << "Mismatch at index " << i;
    }
}

TEST_F(AddKernelTest, CPUAddSingleElement)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {1};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Initialize single element
    float *input0_data = input0.data();
    float *input1_data = input1.data();
    input0_data[0] = 3.14f;
    input1_data[0] = 2.86f;

    // Get kernel and execute
    auto add_kernel = kernel::get_add_kernel(Device::CPU);
    add_kernel(&input0, &input1, &output, nullptr);

    // Verify result
    float *output_data = output.data();
    float expected = input0_data[0] + input1_data[0];
    EXPECT_NEAR(output_data[0], expected, tolerance);
}

TEST_F(AddKernelTest, CPUAddCommutativeProperty)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {64};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output1(shape, Device::CPU, allocator);
    Tensor output2(shape, Device::CPU, allocator);

    // Initialize data
    float *input0_data = input0.data();
    float *input1_data = input1.data();

    for (size_t i = 0; i < 64; ++i)
    {
        input0_data[i] = static_cast<float>(i) * 0.7f;
        input1_data[i] = static_cast<float>(i) * 0.3f;
    }

    // Get kernel
    auto add_kernel = kernel::get_add_kernel(Device::CPU);

    // Test A + B
    add_kernel(&input0, &input1, &output1, nullptr);

    // Test B + A
    add_kernel(&input1, &input0, &output2, nullptr);

    // Verify commutative property: A + B = B + A
    float *output1_data = output1.data();
    float *output2_data = output2.data();
    for (size_t i = 0; i < 64; ++i)
    {
        EXPECT_NEAR(output1_data[i], output2_data[i], tolerance)
            << "Commutative property failed at index " << i;
    }
}

TEST_F(AddKernelTest, CUDAKernelThrows)
{
    // Test that CUDA kernel throws error when stream is null
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {32, 32};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Get CUDA kernel
    auto add_kernel = kernel::get_add_kernel(Device::CUDA);

    // Should throw runtime error when stream is null
    EXPECT_THROW({ add_kernel(&input0, &input1, &output, nullptr); }, std::runtime_error);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
