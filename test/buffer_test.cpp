#include <gtest/gtest.h>
#include "base/buffer.h"
#include "base/allocator.h"
#include "base/tensor.h"
#include <cstring>
#include <vector>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

// ArrBuffer测试
class ArrBufferTestCUDA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        google::InitGoogleLogging("ArrBufferTest");
        FLAGS_logtostderr = true; // Log to stderr for visibility in tests
        allocator = mllm::base::CudaAllocator::getInstance();
    }

    void TearDown() override
    {
        // Clean up resources if needed
        google::ShutdownGoogleLogging();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(ArrBufferTestCUDA, AllocMore)
{
    size_t test_size = 1024 * 1024; // 1MB
    std::vector<mllm::base::Buffer::BufferPtr> vec;
    for (size_t i = 1; i <= 250; i++)
    {
        vec.push_back(mllm::base::Buffer::BufferPtr(new mllm::base::ArrBuffer(allocator, test_size)));
    }

    LOG(INFO) << "alloc 10 GB on CUDA, keep 5 sec...";
    sleep(5);
    LOG(INFO) << "alloc 10 GB on CUDA, keep 5 sec over";

    for (auto &buffer : vec)
    {
        EXPECT_EQ(buffer->size(), test_size);
        EXPECT_NE(buffer->data(), nullptr);
    }
}

TEST_F(ArrBufferTestCUDA, BasicConstruction)
{
    size_t test_size = 1024;
    mllm::base::ArrBuffer buffer(allocator, test_size);

    EXPECT_EQ(buffer.size(), test_size);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(ArrBufferTestCUDA, LargeSizeConstruction)
{
    size_t large_size = 1024 * 1024; // 1MB
    mllm::base::ArrBuffer buffer(allocator, large_size);

    EXPECT_EQ(buffer.size(), large_size);
    EXPECT_NE(buffer.data(), nullptr);
}

// VecBuffer测试
class VecBufferTestCUDA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        allocator = mllm::base::CudaAllocator::getInstance();
        google::InitGoogleLogging("VecBufferTest");
        FLAGS_logtostderr = true; // Log to stderr for visibility in tests
    }

    void TearDown() override
    {
        // Clean up resources if needed
        google::ShutdownGoogleLogging();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(VecBufferTestCUDA, BasicConstruction)
{
    size_t initial_capacity = 100;
    size_t initial_size = 50;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    EXPECT_EQ(buffer.size(), initial_size);
    EXPECT_EQ(buffer.capacity(), initial_capacity);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(VecBufferTestCUDA, ConstructionWithSizeLargerThanCapacity)
{
    size_t initial_capacity = 50;
    size_t initial_size = 100;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    EXPECT_EQ(buffer.size(), initial_size);
    EXPECT_EQ(buffer.capacity(), initial_size); // capacity应该扩展到size
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(VecBufferTestCUDA, ReserveSmallerCapacity)
{
    size_t initial_capacity = 100;
    size_t initial_size = 50;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    size_t smaller_capacity = 30; // 小于当前capacity
    buffer.reserve(smaller_capacity);

    // capacity不应该缩小
    EXPECT_EQ(buffer.capacity(), initial_capacity);
    EXPECT_EQ(buffer.size(), initial_size);
}

// 边界条件和异常处理测试
class BufferExceptionTestCUDA : public ::testing::Test
{
protected:
    void SetUp() override
    {
        allocator = mllm::base::CudaAllocator::getInstance();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(BufferExceptionTestCUDA, ArrBufferMaxSize)
{
    // 测试非常大的size（但不会导致内存分配失败的size）
    size_t large_size = 1024 * 1024 * 10; // 10MB
    try
    {
        mllm::base::ArrBuffer buffer(allocator, large_size);
        EXPECT_EQ(buffer.size(), large_size);
        EXPECT_NE(buffer.data(), nullptr);
    }
    catch (const std::bad_alloc &)
    {
        // 如果内存不足，这是可以接受的
        SUCCEED() << "Memory allocation failed as expected for very large size";
    }
}

TEST_F(BufferExceptionTestCUDA, VecBufferGrowthPattern)
{
    size_t initial_capacity = 4;
    size_t initial_size = 0;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);
    mllm::base::ArrBuffer arr_buffer(allocator, 16);

    // 测试容量增长模式
    std::vector<size_t> capacities;
    capacities.push_back(buffer.capacity());

    // 连续添加数据，观察容量增长
    for (int i = 0; i < 10; ++i)
    {
        buffer.push(arr_buffer.data(), arr_buffer.size());

        size_t current_capacity = buffer.capacity();
        if (current_capacity != capacities.back())
        {
            capacities.push_back(current_capacity);
        }
    }

    // 验证容量是递增的
    for (size_t i = 1; i < capacities.size(); ++i)
    {
        EXPECT_GT(capacities[i], capacities[i - 1]);
    }
}

// ArrBuffer测试
class ArrBufferTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        google::InitGoogleLogging("Qwen3Test");
        FLAGS_logtostderr = true; // Log to stderr for visibility in tests
        allocator = mllm::base::HostAllocator::getInstance();
    }

    void TearDown() override
    {
        // Clean up resources if needed
        google::ShutdownGoogleLogging();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(ArrBufferTest, BasicConstruction)
{
    size_t test_size = 1024;
    mllm::base::ArrBuffer buffer(allocator, test_size);

    EXPECT_EQ(buffer.size(), test_size);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(ArrBufferTest, LargeSizeConstruction)
{
    size_t large_size = 1024 * 1024; // 1MB
    mllm::base::ArrBuffer buffer(allocator, large_size);

    EXPECT_EQ(buffer.size(), large_size);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(ArrBufferTest, DataAccess)
{
    size_t test_size = 100;
    mllm::base::ArrBuffer buffer(allocator, test_size);

    // 写入测试数据
    char *data_ptr = static_cast<char *>(buffer.data());
    for (size_t i = 0; i < test_size; ++i)
    {
        data_ptr[i] = static_cast<char>(i % 256);
    }

    // 验证数据
    for (size_t i = 0; i < test_size; ++i)
    {
        EXPECT_EQ(data_ptr[i], static_cast<char>(i % 256));
    }
}

// VecBuffer测试
class VecBufferTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        allocator = mllm::base::HostAllocator::getInstance();
        google::InitGoogleLogging("Qwen3Test");
        FLAGS_logtostderr = true; // Log to stderr for visibility in tests
    }

    void TearDown() override
    {
        // Clean up resources if needed
        google::ShutdownGoogleLogging();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(VecBufferTest, BasicConstruction)
{
    size_t initial_capacity = 100;
    size_t initial_size = 50;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    EXPECT_EQ(buffer.size(), initial_size);
    EXPECT_EQ(buffer.capacity(), initial_capacity);
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(VecBufferTest, ConstructionWithSizeLargerThanCapacity)
{
    size_t initial_capacity = 50;
    size_t initial_size = 100;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    EXPECT_EQ(buffer.size(), initial_size);
    EXPECT_EQ(buffer.capacity(), initial_size); // capacity应该扩展到size
    EXPECT_NE(buffer.data(), nullptr);
}

TEST_F(VecBufferTest, ConcatWithinCapacity)
{
    size_t initial_capacity = 100;
    size_t initial_size = 20;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    std::string test_data = "Hello World";
    buffer.push(test_data.c_str(), test_data.length());

    EXPECT_EQ(buffer.size(), initial_size + test_data.length());
    EXPECT_EQ(buffer.capacity(), initial_capacity);

    // 验证数据
    char *data_ptr = static_cast<char *>(buffer.data());
    std::string result(data_ptr + initial_size, test_data.length());
    EXPECT_EQ(result, test_data);
}

TEST_F(VecBufferTest, ConcatExceedsCapacity)
{
    size_t initial_capacity = 10;
    size_t initial_size = 5;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    std::string test_data = "This is a long string that exceeds initial capacity";
    size_t old_capacity = buffer.capacity();
    buffer.push(test_data.c_str(), test_data.length());

    EXPECT_EQ(buffer.size(), initial_size + test_data.length());
    EXPECT_GT(buffer.capacity(), old_capacity); // capacity应该增长

    // 验证数据
    char *data_ptr = static_cast<char *>(buffer.data());
    std::string result(data_ptr + initial_size, test_data.length());
    EXPECT_EQ(result, test_data);
}

TEST_F(VecBufferTest, MultipleConcat)
{
    size_t initial_capacity = 100;
    size_t initial_size = 0;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    std::vector<std::string> test_strings = {"Hello", " ", "World", "!", " How", " are", " you?"};
    std::string expected_result;

    for (const auto &str : test_strings)
    {
        buffer.push(str.c_str(), str.length());
        expected_result += str;
    }

    EXPECT_EQ(buffer.size(), expected_result.length());

    // 验证完整数据
    char *data_ptr = static_cast<char *>(buffer.data());
    std::string result(data_ptr, buffer.size());
    EXPECT_EQ(result, expected_result);
}

TEST_F(VecBufferTest, Reserve)
{
    size_t initial_capacity = 50;
    size_t initial_size = 20;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    // 写入一些测试数据
    char *data_ptr = static_cast<char *>(buffer.data());
    for (size_t i = 0; i < initial_size; ++i)
    {
        data_ptr[i] = static_cast<char>('A' + (i % 26));
    }

    size_t new_capacity = 200;
    buffer.reserve(new_capacity);

    EXPECT_EQ(buffer.capacity(), new_capacity);
    EXPECT_EQ(buffer.size(), initial_size); // size应该保持不变

    // 验证数据保持完整
    data_ptr = static_cast<char *>(buffer.data());
    for (size_t i = 0; i < initial_size; ++i)
    {
        EXPECT_EQ(data_ptr[i], static_cast<char>('A' + (i % 26)));
    }
}

TEST_F(VecBufferTest, ReserveSmallerCapacity)
{
    size_t initial_capacity = 100;
    size_t initial_size = 50;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    size_t smaller_capacity = 30; // 小于当前capacity
    buffer.reserve(smaller_capacity);

    // capacity不应该缩小
    EXPECT_EQ(buffer.capacity(), initial_capacity);
    EXPECT_EQ(buffer.size(), initial_size);
}

TEST_F(VecBufferTest, ZeroInitialSize)
{
    size_t initial_capacity = 50;
    size_t initial_size = 0;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.capacity(), initial_capacity);

    // 添加数据
    std::string test_data = "First data";
    buffer.push(test_data.c_str(), test_data.length());

    EXPECT_EQ(buffer.size(), test_data.length());

    char *data_ptr = static_cast<char *>(buffer.data());
    std::string result(data_ptr, buffer.size());
    EXPECT_EQ(result, test_data);
}

// 边界条件和异常处理测试
class BufferExceptionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        allocator = mllm::base::HostAllocator::getInstance();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(BufferExceptionTest, ArrBufferMaxSize)
{
    // 测试非常大的size（但不会导致内存分配失败的size）
    size_t large_size = 1024 * 1024 * 10; // 10MB
    try
    {
        mllm::base::ArrBuffer buffer(allocator, large_size);
        EXPECT_EQ(buffer.size(), large_size);
        EXPECT_NE(buffer.data(), nullptr);
    }
    catch (const std::bad_alloc &)
    {
        // 如果内存不足，这是可以接受的
        SUCCEED() << "Memory allocation failed as expected for very large size";
    }
}

TEST_F(BufferExceptionTest, VecBufferGrowthPattern)
{
    size_t initial_capacity = 4;
    size_t initial_size = 0;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    // 测试容量增长模式
    std::vector<size_t> capacities;
    capacities.push_back(buffer.capacity());

    // 连续添加数据，观察容量增长
    for (int i = 0; i < 10; ++i)
    {
        std::string data = "data" + std::to_string(i) + " ";
        buffer.push(data.c_str(), data.length());

        size_t current_capacity = buffer.capacity();
        if (current_capacity != capacities.back())
        {
            capacities.push_back(current_capacity);
        }
    }

    // 验证容量是递增的
    for (size_t i = 1; i < capacities.size(); ++i)
    {
        EXPECT_GT(capacities[i], capacities[i - 1]);
    }
}

// 性能测试（轻量级）
class BufferPerformanceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        allocator = mllm::base::HostAllocator::getInstance();
    }

    mllm::base::Allocator *allocator;
};

TEST_F(BufferPerformanceTest, VecBufferManySmallConcats)
{
    size_t initial_capacity = 1000;
    size_t initial_size = 0;
    mllm::base::VecBuffer buffer(allocator, initial_capacity, initial_size);

    std::string small_data = "x";
    size_t num_operations = 500;

    for (size_t i = 0; i < num_operations; ++i)
    {
        buffer.push(small_data.c_str(), small_data.length());
    }

    EXPECT_EQ(buffer.size(), num_operations);

    // 验证数据正确性
    char *data_ptr = static_cast<char *>(buffer.data());
    for (size_t i = 0; i < num_operations; ++i)
    {
        EXPECT_EQ(data_ptr[i], 'x');
    }
}

TEST_F(BufferPerformanceTest, ArrBufferLargeDataAccess)
{
    size_t large_size = 1024 * 100; // 100KB
    mllm::base::ArrBuffer buffer(allocator, large_size);

    // 写入模式数据
    char *data_ptr = static_cast<char *>(buffer.data());
    for (size_t i = 0; i < large_size; ++i)
    {
        data_ptr[i] = static_cast<char>(i % 256);
    }

    // 验证数据完整性
    for (size_t i = 0; i < large_size; ++i)
    {
        EXPECT_EQ(data_ptr[i], static_cast<char>(i % 256));
    }
}
