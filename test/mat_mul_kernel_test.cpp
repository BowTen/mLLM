#include "kernel/kernel.h"
#include "base/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#include <cuda_runtime.h>

using namespace mllm;
using namespace mllm::base;

class MatMulKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test parameters
        tolerance = 1e-5f;

        // Initialize random number generator for consistent tests
        rng.seed(42);
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    void TearDown() override
    {
        // Clean up if needed
    }

    // Helper function to fill tensor with random values
    void fillTensorRandom(Tensor &tensor)
    {
        float *data = tensor.data();
        size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = dist(rng);
        }
    }

    // Helper function to fill tensor with specific values
    void fillTensorSequential(Tensor &tensor, float start = 1.0f, float step = 1.0f)
    {
        float *data = tensor.data();
        size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = start + static_cast<float>(i) * step;
        }
    }

    // Helper function to verify matrix multiplication manually
    void verifyMatMul(const Tensor &input0, const Tensor &input1, const Tensor &output)
    {
        auto shape0 = input0.shape();
        auto shape1 = input1.shape();
        auto output_shape = output.shape();

        size_t M = shape0[0]; // rows of first matrix
        size_t K = shape0[1]; // cols of first matrix / rows of second matrix
        size_t N = shape1[1]; // cols of second matrix

        const float *data0 = const_cast<Tensor &>(input0).data();
        const float *data1 = const_cast<Tensor &>(input1).data();
        const float *output_data = const_cast<Tensor &>(output).data();

        for (size_t i = 0; i < M; ++i)
        {
            for (size_t j = 0; j < N; ++j)
            {
                float expected = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    expected += data0[i * K + k] * data1[k * N + j];
                }
                float actual = output_data[i * N + j];
                EXPECT_NEAR(actual, expected, tolerance)
                    << "Mismatch at position (" << i << ", " << j << "), expected: "
                    << expected << ", actual: " << actual;
            }
        }
    }

    float tolerance;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
};

TEST_F(MatMulKernelTest, CPUMatMulSquareMatrices)
{
    auto allocator = HostAllocator::getInstance();

    // Test 4x4 * 4x4 matrix multiplication
    std::vector<size_t> shape = {4, 4};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor input1(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Fill with sequential values for easy verification
    fillTensorSequential(input0, 1.0f, 1.0f);
    fillTensorSequential(input1, 0.1f, 0.1f);

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    verifyMatMul(input0, input1, output);
}

TEST_F(MatMulKernelTest, CPUMatMulRectangularMatrices)
{
    auto allocator = HostAllocator::getInstance();

    // Test 3x5 * 5x2 matrix multiplication
    std::vector<size_t> shape0 = {3, 5};
    std::vector<size_t> shape1 = {5, 2};
    std::vector<size_t> output_shape = {3, 2};

    Tensor input0(shape0, Device::CPU, allocator);
    Tensor input1(shape1, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill with sequential values
    fillTensorSequential(input0, 1.0f, 0.5f);
    fillTensorSequential(input1, 2.0f, 0.2f);

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    verifyMatMul(input0, input1, output);
}

TEST_F(MatMulKernelTest, CPUMatMulIdentityMatrix)
{
    auto allocator = HostAllocator::getInstance();

    // Test matrix * identity matrix
    std::vector<size_t> shape = {3, 3};
    Tensor input0(shape, Device::CPU, allocator);
    Tensor identity(shape, Device::CPU, allocator);
    Tensor output(shape, Device::CPU, allocator);

    // Fill input0 with random values
    fillTensorRandom(input0);

    // Create identity matrix
    float *identity_data = identity.data();
    for (size_t i = 0; i < 9; ++i)
        identity_data[i] = 0.0f;
    identity_data[0] = identity_data[4] = identity_data[8] = 1.0f; // diagonal elements

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &identity, &output, nullptr);

    // Result should be the same as input0
    float *input0_data = input0.data();
    float *output_data = output.data();
    for (size_t i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(output_data[i], input0_data[i], tolerance)
            << "Mismatch at index " << i;
    }
}

TEST_F(MatMulKernelTest, CPUMatMulZeroMatrix)
{
    auto allocator = HostAllocator::getInstance();

    std::vector<size_t> shape = {2, 3};
    std::vector<size_t> zero_shape = {3, 4};
    std::vector<size_t> output_shape = {2, 4};

    Tensor input0(shape, Device::CPU, allocator);
    Tensor zero_matrix(zero_shape, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill input0 with random values
    fillTensorRandom(input0);

    // Create zero matrix
    float *zero_data = zero_matrix.data();
    for (size_t i = 0; i < zero_matrix.size(); ++i)
    {
        zero_data[i] = 0.0f;
    }

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &zero_matrix, &output, nullptr);

    // Result should be all zeros
    float *output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_NEAR(output_data[i], 0.0f, tolerance)
            << "Mismatch at index " << i;
    }
}

TEST_F(MatMulKernelTest, CPUMatMulLargeMatrices)
{
    auto allocator = HostAllocator::getInstance();

    // Test larger matrices: 64x32 * 32x64
    std::vector<size_t> shape0 = {64, 32};
    std::vector<size_t> shape1 = {32, 64};
    std::vector<size_t> output_shape = {64, 64};

    Tensor input0(shape0, Device::CPU, allocator);
    Tensor input1(shape1, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill with small random values to avoid overflow
    std::uniform_real_distribution<float> small_dist(-0.1f, 0.1f);
    float *data0 = input0.data();
    float *data1 = input1.data();
    for (size_t i = 0; i < input0.size(); ++i)
    {
        data0[i] = small_dist(rng);
    }
    for (size_t i = 0; i < input1.size(); ++i)
    {
        data1[i] = small_dist(rng);
    }

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Verify a few sample positions manually
    auto shape0_vec = input0.shape();
    auto shape1_vec = input1.shape();
    size_t K = shape0_vec[1];
    size_t N = shape1_vec[1];

    float *output_data = output.data();

    // Check a few positions
    std::vector<std::pair<size_t, size_t>> check_positions = {{0, 0}, {10, 20}, {30, 40}, {63, 63}};

    for (auto pos : check_positions)
    {
        size_t i = pos.first;
        size_t j = pos.second;
        float expected = 0.0f;
        for (size_t k = 0; k < K; ++k)
        {
            expected += data0[i * K + k] * data1[k * N + j];
        }
        float actual = output_data[i * N + j];
        EXPECT_NEAR(actual, expected, tolerance)
            << "Mismatch at position (" << i << ", " << j << ")";
    }
}

TEST_F(MatMulKernelTest, CPUMatMulSingleRowColumn)
{
    auto allocator = HostAllocator::getInstance();

    // Test 1x5 * 5x1 matrix multiplication (results in 1x1)
    std::vector<size_t> shape0 = {1, 5};
    std::vector<size_t> shape1 = {5, 1};
    std::vector<size_t> output_shape = {1, 1};

    Tensor input0(shape0, Device::CPU, allocator);
    Tensor input1(shape1, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill with known values
    float *data0 = input0.data();
    float *data1 = input1.data();
    for (size_t i = 0; i < 5; ++i)
    {
        data0[i] = static_cast<float>(i + 1);        // [1, 2, 3, 4, 5]
        data1[i] = 0.1f * static_cast<float>(i + 1); // [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Expected result: 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 + 5*0.5 = 5.5
    float expected = 1 * 0.1f + 2 * 0.2f + 3 * 0.3f + 4 * 0.4f + 5 * 0.5f;
    float *output_data = output.data();
    EXPECT_NEAR(output_data[0], expected, tolerance);
}

TEST_F(MatMulKernelTest, CPUMatMulVectorMultiplication)
{
    auto allocator = HostAllocator::getInstance();

    // Test 5x1 * 1x3 matrix multiplication (outer product, results in 5x3)
    std::vector<size_t> shape0 = {5, 1};
    std::vector<size_t> shape1 = {1, 3};
    std::vector<size_t> output_shape = {5, 3};

    Tensor input0(shape0, Device::CPU, allocator);
    Tensor input1(shape1, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill with known values
    float *data0 = input0.data();
    float *data1 = input1.data();

    // Vector [1, 2, 3, 4, 5]^T
    for (size_t i = 0; i < 5; ++i)
    {
        data0[i] = static_cast<float>(i + 1);
    }

    // Vector [2, 3, 4]
    data1[0] = 2.0f;
    data1[1] = 3.0f;
    data1[2] = 4.0f;

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Verify outer product result
    float *output_data = output.data();
    for (size_t i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            float expected = data0[i] * data1[j];
            float actual = output_data[i * 3 + j];
            EXPECT_NEAR(actual, expected, tolerance)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}

TEST_F(MatMulKernelTest, CPUMatMulNonSquareToSquare)
{
    auto allocator = HostAllocator::getInstance();

    // Test 2x6 * 6x2 matrix multiplication (results in 2x2)
    std::vector<size_t> shape0 = {2, 6};
    std::vector<size_t> shape1 = {6, 2};
    std::vector<size_t> output_shape = {2, 2};

    Tensor input0(shape0, Device::CPU, allocator);
    Tensor input1(shape1, Device::CPU, allocator);
    Tensor output(output_shape, Device::CPU, allocator);

    // Fill with sequential values
    fillTensorSequential(input0, 1.0f, 1.0f);
    fillTensorSequential(input1, 0.1f, 0.1f);

    // Get kernel and execute
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);
    matmul_kernel(&input0, &input1, &output, nullptr);

    // Verify results
    verifyMatMul(input0, input1, output);
}

TEST_F(MatMulKernelTest, CPUMatMulAssociativity)
{
    auto allocator = HostAllocator::getInstance();

    // Test associativity property: (A * B) * C = A * (B * C)
    // Use smaller matrices for easier computation
    std::vector<size_t> shapeA = {2, 3};
    std::vector<size_t> shapeB = {3, 2};
    std::vector<size_t> shapeC = {2, 2};
    std::vector<size_t> shapeAB = {2, 2};
    std::vector<size_t> shapeBC = {3, 2};

    Tensor A(shapeA, Device::CPU, allocator);
    Tensor B(shapeB, Device::CPU, allocator);
    Tensor C(shapeC, Device::CPU, allocator);
    Tensor AB(shapeAB, Device::CPU, allocator);
    Tensor BC(shapeBC, Device::CPU, allocator);
    Tensor AB_C(shapeC, Device::CPU, allocator); // (A*B)*C
    Tensor A_BC(shapeC, Device::CPU, allocator); // A*(B*C)

    // Fill with small values to avoid numerical precision issues
    fillTensorSequential(A, 0.1f, 0.1f);
    fillTensorSequential(B, 0.2f, 0.1f);
    fillTensorSequential(C, 0.3f, 0.1f);

    // Get kernel
    auto matmul_kernel = kernel::get_matmul_kernel(Device::CPU);

    // Compute A * B
    matmul_kernel(&A, &B, &AB, nullptr);

    // Compute B * C
    matmul_kernel(&B, &C, &BC, nullptr);

    // Compute (A * B) * C
    matmul_kernel(&AB, &C, &AB_C, nullptr);

    // Compute A * (B * C)
    matmul_kernel(&A, &BC, &A_BC, nullptr);

    // Verify associativity: (A * B) * C = A * (B * C)
    float *data1 = AB_C.data();
    float *data2 = A_BC.data();
    for (size_t i = 0; i < 4; ++i) // 2x2 = 4 elements
    {
        EXPECT_NEAR(data1[i], data2[i], tolerance * 10.0f) // Higher tolerance for accumulated errors
            << "Associativity failed at index " << i
            << ", (A*B)*C = " << data1[i] << ", A*(B*C) = " << data2[i];
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
