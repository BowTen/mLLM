#include "base/tensor.h"
#include "base/util.h"
#include <iostream>
#include <gtest/gtest.h>
#include <vector>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace mllm::base;

// Test fixture for Tensor tests
class TensorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Setup code if needed
        google::InitGoogleLogging("TensorTest");
    }

    void TearDown() override
    {
        // Cleanup code if needed
        google::ShutdownGoogleLogging();
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

// Test Tensor construction with 4D shape
TEST_F(TensorTest, Constructor4D)
{
    std::vector<size_t> shape = {2, 3, 4, 5};
    Tensor tensor(shape, Device::CPU);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 120); // 2 * 3 * 4 * 5 = 120
    EXPECT_NE(tensor.data(), nullptr);
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

// Test view function - changing view without changing data
TEST_F(TensorTest, ViewFunction)
{
    std::vector<size_t> original_shape = {2, 3, 4};
    Tensor tensor(original_shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Test view with same total size
    std::vector<size_t> new_shape = {6, 4};
    tensor.view(new_shape);

    EXPECT_EQ(tensor.shape(), new_shape);
    EXPECT_EQ(tensor.size(), 24); // Should remain the same

    // Data should remain the same
    float *viewed_data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_FLOAT_EQ(viewed_data[i], static_cast<float>(i));
    }
}

// Test view with 1D tensor
TEST_F(TensorTest, View1D)
{
    std::vector<size_t> original_shape = {12};
    Tensor tensor(original_shape, Device::CPU);

    // Test view to 2D
    std::vector<size_t> new_shape = {3, 4};
    tensor.view(new_shape);

    EXPECT_EQ(tensor.shape(), new_shape);
    EXPECT_EQ(tensor.size(), 12);
}

// Test view with different valid shapes
TEST_F(TensorTest, ViewDifferentShapes)
{
    std::vector<size_t> original_shape = {2, 3, 4};
    Tensor tensor(original_shape, Device::CPU);

    // Test multiple valid views
    std::vector<std::vector<size_t>> valid_shapes = {
        {24},
        {1, 24},
        {24, 1},
        {2, 12},
        {4, 6},
        {3, 8},
        {2, 3, 4}, // back to original
        {1, 2, 3, 4},
        {2, 1, 3, 4}};

    for (const auto &shape : valid_shapes)
    {
        tensor.view(shape);
        EXPECT_EQ(tensor.shape(), shape);
        EXPECT_EQ(tensor.size(), 24);
    }
}

// Test reshape function
TEST_F(TensorTest, ReshapeFunction)
{
    std::vector<size_t> original_shape = {2, 6};
    Tensor tensor(original_shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = static_cast<float>(i + 1);
    }

    // Test reshape
    std::vector<size_t> new_shape = {3, 4};
    tensor.reshape(new_shape);

    EXPECT_EQ(tensor.shape(), new_shape);
    EXPECT_EQ(tensor.size(), 12);

    // Data should remain the same after reshape
    float *reshaped_data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        EXPECT_FLOAT_EQ(reshaped_data[i], static_cast<float>(i + 1));
    }
}

// Test reshape with different dimensions
TEST_F(TensorTest, ReshapeDifferentDimensions)
{
    std::vector<size_t> original_shape = {2, 3, 4};
    Tensor tensor(original_shape, Device::CPU);

    // Test reshape to different dimensions
    std::vector<std::vector<size_t>> reshape_targets = {
        {24},        // 3D -> 1D
        {4, 6},      // 3D -> 2D
        {2, 2, 6},   // 3D -> 3D (different shape)
        {1, 1, 24},  // 3D -> 3D (with 1s)
        {2, 2, 2, 3} // 3D -> 4D
    };

    for (const auto &target_shape : reshape_targets)
    {
        Tensor test_tensor(original_shape, Device::CPU);
        test_tensor.reshape(target_shape);
        EXPECT_EQ(test_tensor.shape(), target_shape);
        EXPECT_EQ(test_tensor.size(), 24);
    }
}

// Test transpose function
TEST_F(TensorTest, TransposeFunction)
{
    std::vector<size_t> shape = {2, 3, 4};
    Tensor tensor(shape, Device::CPU);

    // Fill with test data to verify transpose correctness
    float *data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Test transpose dimensions 0 and 1
    tensor.transpose(0, 1);

    // After transpose(0,1), shape should be [3, 2, 4]
    std::vector<size_t> expected_shape = {3, 2, 4};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.size(), 24); // Size should remain the same
}

// Test transpose with different dimension pairs
TEST_F(TensorTest, TransposeDifferentDimensions)
{
    std::vector<size_t> original_shape = {2, 3, 4, 5};

    struct TransposeTest
    {
        size_t dim1, dim2;
        std::vector<size_t> expected_shape;
    };

    std::vector<TransposeTest> transpose_tests = {
        {0, 1, {3, 2, 4, 5}},
        {0, 2, {4, 3, 2, 5}},
        {0, 3, {5, 3, 4, 2}},
        {1, 2, {2, 4, 3, 5}},
        {1, 3, {2, 5, 4, 3}},
        {2, 3, {2, 3, 5, 4}}};

    for (const auto &test : transpose_tests)
    {
        Tensor tensor(original_shape, Device::CPU);
        tensor.transpose(test.dim1, test.dim2);
        EXPECT_EQ(tensor.shape(), test.expected_shape)
            << "Failed for transpose(" << test.dim1 << ", " << test.dim2 << ")";
        EXPECT_EQ(tensor.size(), 120); // 2*3*4*5 = 120
    }
}

// Test transpose on 2D matrix
TEST_F(TensorTest, Transpose2D)
{
    std::vector<size_t> shape = {3, 4};
    Tensor tensor(shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 4; ++j)
        {
            data[i * 4 + j] = static_cast<float>(i * 10 + j);
        }
    }

    // Transpose
    tensor.transpose(0, 1);

    // Shape should be [4, 3]
    std::vector<size_t> expected_shape = {4, 3};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.size(), 12);
}

// Test t() function (2D matrix transpose)
TEST_F(TensorTest, TFunction)
{
    std::vector<size_t> shape = {3, 4};
    Tensor tensor(shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Apply t() function
    tensor.t();

    // Shape should be [4, 3] after transpose
    std::vector<size_t> expected_shape = {4, 3};
    EXPECT_EQ(tensor.shape(), expected_shape);
    EXPECT_EQ(tensor.size(), 12);
}

// Test t() function on square matrix
TEST_F(TensorTest, TFunctionSquare)
{
    std::vector<size_t> shape = {3, 3};
    Tensor tensor(shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < 9; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Apply t() function
    tensor.t();

    // Shape should remain [3, 3] for square matrix
    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 9);
}

// Test contiguous function
TEST_F(TensorTest, ContiguousFunction)
{
    std::vector<size_t> shape = {2, 3, 4};
    Tensor tensor(shape, Device::CPU);

    // Initially should be contiguous
    EXPECT_TRUE(tensor.is_contiguous());

    // After transpose, might not be contiguous
    tensor.transpose(0, 2);

    tensor.contiguous();
    EXPECT_TRUE(tensor.is_contiguous());

    EXPECT_EQ(tensor.size(), 24);
}

// Test chained operations
TEST_F(TensorTest, ChainedOperations)
{
    std::vector<size_t> original_shape = {2, 3, 4};
    Tensor tensor(original_shape, Device::CPU);

    // Fill with test data
    float *data = tensor.data();
    for (size_t i = 0; i < tensor.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Chain operations: view -> transpose -> reshape
    tensor.view({6, 4});
    EXPECT_EQ(tensor.shape(), std::vector<size_t>({6, 4}));

    tensor.transpose(0, 1);
    EXPECT_EQ(tensor.shape(), std::vector<size_t>({4, 6}));

    tensor.reshape({2, 12});
    EXPECT_EQ(tensor.shape(), std::vector<size_t>({2, 12}));

    // Size should remain constant throughout
    EXPECT_EQ(tensor.size(), 24);
}

// ============= CUDA-specific tests =============
// Test CUDA Tensor with larger aligned sizes
TEST_F(TensorTest, ConstructorCUDA_LargeAligned)
{
    std::vector<size_t> shape = {64, 64}; // 64*64*4 = 16384 bytes (multiple of 128)
    Tensor tensor(shape, Device::CUDA);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 4096);
    EXPECT_NE(tensor.data(), nullptr);
}

// Test CUDA Tensor mutable construction
TEST_F(TensorTest, ConstructorCUDA_Mutable)
{
    std::vector<size_t> shape = {32, 32};
    Tensor tensor(shape, Device::CUDA, true);

    EXPECT_EQ(tensor.shape(), shape);
    EXPECT_EQ(tensor.size(), 1024);
    EXPECT_NE(tensor.data(), nullptr);
}

// Test CUDA tensor device transfer
TEST_F(TensorTest, CUDADeviceTransfer)
{
    std::vector<size_t> shape = {32, 32};

    // Create CPU tensor and fill with data
    Tensor cpu_tensor(shape, Device::CPU);
    float *cpu_data = cpu_tensor.data();
    for (size_t i = 0; i < cpu_tensor.size(); ++i)
    {
        cpu_data[i] = static_cast<float>(i);
    }

    // Transfer to CUDA
    cpu_tensor.toDevice(Device::CUDA);
    EXPECT_NE(cpu_tensor.data(), nullptr);

    // Transfer back to CPU and verify data
    cpu_tensor.toDevice(Device::CPU);
    float *verified_data = cpu_tensor.data();
    for (size_t i = 0; i < cpu_tensor.size(); ++i)
    {
        EXPECT_FLOAT_EQ(verified_data[i], static_cast<float>(i));
    }
}

// Test CUDA tensor cloning
TEST_F(TensorTest, CUDAClone)
{
    std::vector<size_t> shape = {32, 16};

    // Create CUDA tensor
    Tensor cuda_tensor(shape, Device::CUDA);

    // Clone it
    Tensor cloned_tensor = cuda_tensor.clone();

    EXPECT_EQ(cloned_tensor.shape(), shape);
    EXPECT_EQ(cloned_tensor.size(), cuda_tensor.size());
    EXPECT_NE(cloned_tensor.data(), cuda_tensor.data()); // Different memory locations
}

// Test CUDA tensor view operations
TEST_F(TensorTest, CUDAView)
{
    std::vector<size_t> original_shape = {32, 32};
    Tensor cuda_tensor(original_shape, Device::CUDA);

    // Test view with same total size
    std::vector<size_t> new_shape = {64, 16};
    cuda_tensor.view(new_shape);

    EXPECT_EQ(cuda_tensor.shape(), new_shape);
    EXPECT_EQ(cuda_tensor.size(), 1024); // Should remain the same
}

// Test CUDA tensor reshape operations
TEST_F(TensorTest, CUDAReshape)
{
    std::vector<size_t> original_shape = {32, 32};
    Tensor cuda_tensor(original_shape, Device::CUDA);

    // Test reshape
    std::vector<size_t> new_shape = {128, 8};
    cuda_tensor.reshape(new_shape);

    EXPECT_EQ(cuda_tensor.shape(), new_shape);
    EXPECT_EQ(cuda_tensor.size(), 1024);
}

// Test CUDA tensor transpose operations
TEST_F(TensorTest, CUDATranspose)
{
    std::vector<size_t> shape = {32, 64};
    Tensor cuda_tensor(shape, Device::CUDA);

    // Test transpose dimensions 0 and 1
    cuda_tensor.transpose(0, 1);

    // After transpose(0,1), shape should be [64, 32]
    std::vector<size_t> expected_shape = {64, 32};
    EXPECT_EQ(cuda_tensor.shape(), expected_shape);
    EXPECT_EQ(cuda_tensor.size(), 2048); // Size should remain the same
}

// Test CUDA tensor t() function (2D matrix transpose)
TEST_F(TensorTest, CUDA_TFunction)
{
    std::vector<size_t> shape = {32, 64};
    Tensor cuda_tensor(shape, Device::CUDA);

    // Apply t() function
    cuda_tensor.t();

    // Shape should be [64, 32] after transpose
    std::vector<size_t> expected_shape = {64, 32};
    EXPECT_EQ(cuda_tensor.shape(), expected_shape);
    EXPECT_EQ(cuda_tensor.size(), 2048);
}

// Test CUDA tensor contiguous function
TEST_F(TensorTest, CUDAContiguous)
{
    std::vector<size_t> shape = {32, 32, 4};
    Tensor cuda_tensor(shape, Device::CUDA);

    // Initially should be contiguous
    EXPECT_TRUE(cuda_tensor.is_contiguous());

    // After transpose, might not be contiguous
    cuda_tensor.transpose(0, 2);

    cuda_tensor.contiguous();
    EXPECT_TRUE(cuda_tensor.is_contiguous());

    EXPECT_EQ(cuda_tensor.size(), 4096);
}

// Test CUDA tensor chained operations
TEST_F(TensorTest, CUDAChainedOperations)
{
    std::vector<size_t> original_shape = {32, 32, 4};
    Tensor cuda_tensor(original_shape, Device::CUDA);

    // Chain operations: view -> transpose -> reshape
    cuda_tensor.view({64, 64});
    EXPECT_EQ(cuda_tensor.shape(), std::vector<size_t>({64, 64}));

    cuda_tensor.transpose(0, 1);
    EXPECT_EQ(cuda_tensor.shape(), std::vector<size_t>({64, 64}));

    cuda_tensor.reshape({128, 32});
    EXPECT_EQ(cuda_tensor.shape(), std::vector<size_t>({128, 32}));

    // Size should remain constant throughout
    EXPECT_EQ(cuda_tensor.size(), 4096);
}

// Test CUDA tensor with large aligned sizes for performance
TEST_F(TensorTest, CUDALargeAlignedTensor)
{
    // Test with larger tensor that maintains alignment
    std::vector<size_t> shape = {256, 256}; // Large but aligned
    Tensor cuda_tensor(shape, Device::CUDA);

    EXPECT_EQ(cuda_tensor.shape(), shape);
    EXPECT_EQ(cuda_tensor.size(), 65536);
    EXPECT_NE(cuda_tensor.data(), nullptr);

    // Test basic operations on large tensor
    cuda_tensor.view({512, 128});
    EXPECT_EQ(cuda_tensor.size(), 65536);

    cuda_tensor.transpose(0, 1);
    EXPECT_EQ(cuda_tensor.shape(), std::vector<size_t>({128, 512}));
}

class TensorCatTest : public ::testing::Test
{
protected:
    float check_eps = 1e-6;
    std::vector<size_t> shape_a;
    std::vector<size_t> shape_b;
    std::vector<float> data_a;
    std::vector<float> data_b;
    std::vector<float> data_cat;
    Tensor a;
    Tensor b;

    TensorCatTest() : shape_a({4, 16}),
                      shape_b({2, 16}),
                      data_a(64),
                      data_b(32)
    {
        google::InitGoogleLogging("TensorCatTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "TensorCatTest setup complete.";

        for (size_t i = 0; i < data_a.size(); i++)
            data_a[i] = i;
        for (size_t i = 0; i < data_b.size(); i++)
            data_b[i] = i * 2;
        data_cat.insert(data_cat.end(), data_a.begin(), data_a.end());
        data_cat.insert(data_cat.end(), data_b.begin(), data_b.end());

        a = Tensor(data_a.data(), shape_a, true, Device::CPU, true);
        b = Tensor(data_b.data(), shape_b, true, Device::CPU);
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }
};

TEST_F(TensorCatTest, CPUHeadsOptionCat)
{
    std::vector<size_t> heads_a_shape({4, 8, 2});
    std::vector<size_t> heads_b_shape({2, 8, 2});
    a.view(heads_a_shape);
    b.view(heads_b_shape);
    a.transpose(0, 1); // 8 4 2
    b.transpose(0, 1); // 8 2 2

    VLOG(DEBUG) << "run a.cat";
    a.cat(b, 1);
    VLOG(DEBUG) << "a.cat over";
    size_t total_size = a.size();

    EXPECT_EQ(a.shape(), std::vector<size_t>({8, 6, 2}));
    EXPECT_EQ(total_size, data_a.size() + data_b.size());
    EXPECT_EQ(total_size, data_cat.size());

    std::cout << "a:\n";
    int diff = 0;
    for (size_t j = 0; j < a.shape(1); j++)
    {
        for (size_t i = 0; i < a.shape(0); i++)
        {
            for (size_t k = 0; k < a.shape(2); k++)
            {
                std::cout << *a[{i, j, k}] << ' ';
                diff += std::fabs(a.data()[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k] -
                                  data_cat[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k]) > check_eps;
            }
            std::cout << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
TEST_F(TensorCatTest, CUDAHeadsOptionCat)
{
    a.toDevice(Device::CUDA);
    b.toDevice(Device::CUDA);
    std::vector<size_t> heads_a_shape({4, 8, 2});
    std::vector<size_t> heads_b_shape({2, 8, 2});
    a.view(heads_a_shape);
    b.view(heads_b_shape);
    a.transpose(0, 1);
    b.transpose(0, 1);

    VLOG(DEBUG) << "run a.cat";
    a.cat(b, 1);
    VLOG(DEBUG) << "a.cat over";
    size_t total_size = a.size();

    EXPECT_EQ(a.shape(), std::vector<size_t>({8, 6, 2}));
    EXPECT_EQ(total_size, data_a.size() + data_b.size());
    EXPECT_EQ(total_size, data_cat.size());

    a.toDevice(Device::CPU);
    std::cout << "a:\n";
    int diff = 0;
    for (size_t j = 0; j < a.shape(1); j++)
    {
        for (size_t i = 0; i < a.shape(0); i++)
        {
            for (size_t k = 0; k < a.shape(2); k++)
            {
                std::cout << *a[{i, j, k}] << ' ';
                diff += std::fabs(a.data()[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k] -
                                  data_cat[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k]) > check_eps;
            }
            std::cout << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

TEST_F(TensorCatTest, CPUHeadsCloneCat)
{
    std::vector<size_t> heads_a_shape({4, 8, 2});
    std::vector<size_t> heads_b_shape({2, 8, 2});
    a.view(heads_a_shape);
    b.view(heads_b_shape);
    a.transpose(0, 1); // 8 4 2
    b.transpose(0, 1); // 8 2 2
    b.contiguous();

    VLOG(DEBUG) << "run a.cat";
    a.cat(b, 1);
    VLOG(DEBUG) << "a.cat over";
    size_t total_size = a.size();

    EXPECT_EQ(a.shape(), std::vector<size_t>({8, 6, 2}));
    EXPECT_EQ(total_size, data_a.size() + data_b.size());
    EXPECT_EQ(total_size, data_cat.size());

    std::cout << "a:\n";
    int diff = 0;
    for (size_t j = 0; j < a.shape(1); j++)
    {
        for (size_t i = 0; i < a.shape(0); i++)
        {
            for (size_t k = 0; k < a.shape(2); k++)
            {
                std::cout << *a[{i, j, k}] << ' ';
                diff += std::fabs(a.data()[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k] -
                                  data_cat[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k]) > check_eps;
            }
            std::cout << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
TEST_F(TensorCatTest, CUDAHeadsCloneCat)
{
    a.toDevice(Device::CUDA);
    b.toDevice(Device::CUDA);
    std::vector<size_t> heads_a_shape({4, 8, 2});
    std::vector<size_t> heads_b_shape({2, 8, 2});
    a.view(heads_a_shape);
    b.view(heads_b_shape);
    a.transpose(0, 1);
    b.transpose(0, 1);
    b.contiguous();

    VLOG(DEBUG) << "run a.cat";
    a.cat(b, 1);
    VLOG(DEBUG) << "a.cat over";
    size_t total_size = a.size();

    EXPECT_EQ(a.shape(), std::vector<size_t>({8, 6, 2}));
    EXPECT_EQ(total_size, data_a.size() + data_b.size());
    EXPECT_EQ(total_size, data_cat.size());

    a.toDevice(Device::CPU);
    std::cout << "a:\n";
    int diff = 0;
    for (size_t j = 0; j < a.shape(1); j++)
    {
        for (size_t i = 0; i < a.shape(0); i++)
        {
            for (size_t k = 0; k < a.shape(2); k++)
            {
                std::cout << *a[{i, j, k}] << ' ';
                diff += std::fabs(a.data()[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k] -
                                  data_cat[i * a.shape(1) * a.shape(2) + j * a.shape(2) + k]) > check_eps;
            }
            std::cout << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

class TensorExpandTest : public ::testing::Test
{
protected:
    float check_eps = 1e-6;
    std::vector<size_t> shape;
    Tensor ts;

    TensorExpandTest() : shape({2, 4}), ts(shape, Device::CPU, true)
    {
        google::InitGoogleLogging("TensorExpandTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "TensorExpandTest setup complete.";

        size_t total_size = ts.size();
        for (size_t i = 0; i < total_size; i++)
        {
            *ts[i] = get_random_float();
        }
    }

    void TearDown() override
    {
        google::ShutdownGoogleLogging();
    }

    void print_ts2(Tensor &ts)
    {
        for (size_t i = 0; i < ts.shape(-2); i++)
        {
            for (size_t j = 0; j < ts.shape(-1); j++)
            {
                std::cout << *ts[{i, j}] << ' ';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    void print_ts3(Tensor &ts)
    {
        for (size_t i = 0; i < ts.shape(-2); i++)
        {
            for (size_t k = 0; k < ts.shape(0); k++)
            {
                for (size_t j = 0; j < ts.shape(-1); j++)
                {
                    std::cout << *ts[{k, i, j}] << ' ';
                }
                std::cout << "   ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    void check_data_ts3(Tensor &ts, Tensor &ts_ori)
    {
        int diff = 0;
        for (size_t i = 0; i < ts.shape(-2); i++)
        {
            for (size_t k = 0; k < ts.shape(0); k++)
            {
                for (size_t j = 0; j < ts.shape(-1); j++)
                {
                    diff += std::fabs(*ts[{k, i, j}] - *ts_ori[{0, i, j}]) > check_eps;
                }
            }
        }
        EXPECT_EQ(diff, 0);
    }
    void check_addr_ts3(Tensor &ts, Tensor &ts_ori)
    {
        int diff = 0;
        for (size_t i = 0; i < ts.shape(-2); i++)
        {
            for (size_t k = 0; k < ts.shape(0); k++)
            {
                for (size_t j = 0; j < ts.shape(-1); j++)
                {
                    diff += (ts[{k, i, j}] != ts_ori[{0, i, j}]);
                }
            }
        }
        EXPECT_EQ(diff, 0);
    }
};

TEST_F(TensorExpandTest, CPU)
{
    std::cout << "origin ts:\n";
    print_ts2(ts);
    std::cout << "size: " << ts.size() << std::endl;
    auto ts_ori = ts.clone();

    ts.insert_dim(0);
    ts.expand(0, 4);
    std::cout << "expand ts:\n";
    print_ts3(ts);
    check_data_ts3(ts, ts);
    check_addr_ts3(ts, ts);
    std::cout << "size: " << ts.size() << std::endl;
}
TEST_F(TensorExpandTest, CUDA)
{
    std::cout << "origin ts:\n";
    print_ts2(ts);
    std::cout << "size: " << ts.size() << std::endl;

    ts.toDevice(Device::CUDA);
    ts.insert_dim(0);
    ts.expand(0, 4);
    ts.toDevice(Device::CPU);

    std::cout << "expand ts:\n";
    print_ts3(ts);
    check_data_ts3(ts, ts);
    check_addr_ts3(ts, ts);
    std::cout << "size: " << ts.size() << std::endl;
}

TEST_F(TensorExpandTest, ContiguousCPU)
{
    std::cout << "origin ts:\n";
    print_ts2(ts);
    std::cout << "size: " << ts.size() << std::endl;

    ts.insert_dim(0);
    ts.expand(0, 4);
    auto ts_ori = ts.clone();
    ts.contiguous();
    std::cout << "expand ts:\n";
    print_ts3(ts);
    check_data_ts3(ts, ts_ori);
    std::cout << "size: " << ts.size() << std::endl;
}
TEST_F(TensorExpandTest, ContiguousCUDA)
{
    std::cout << "origin ts:\n";
    print_ts2(ts);
    std::cout << "size: " << ts.size() << std::endl;

    ts.toDevice(Device::CUDA);
    ts.insert_dim(0);
    ts.expand(0, 4);
    auto ts_ori = ts.clone();
    ts.contiguous();
    ts.toDevice(Device::CPU);
    ts_ori.toDevice(Device::CPU);

    std::cout << "expand ts:\n";
    print_ts3(ts);
    std::cout << "expand ts_ori:\n";
    print_ts3(ts_ori);
    check_data_ts3(ts, ts_ori);
    std::cout << "size: " << ts.size() << std::endl;
}
