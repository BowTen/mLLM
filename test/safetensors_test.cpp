#include "base/safetensors.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace mllm::base;

// Test fixture for SafeTensors tests
class SafeTensorsTest : public ::testing::Test
{
protected:
    std::string test_file_path;

    void SetUp() override
    {
        test_file_path = "/tmp/test_safetensors.bin";
    }

    void TearDown() override
    {
        // Clean up test file
        std::remove(test_file_path.c_str());
    }

    // Helper function to create a test safetensors file
    bool create_test_safetensors_file(const std::string &file_path)
    {
        // Create test header
        json header;
        header["test_weight_1"] = {
            {"dtype", "F32"},
            {"shape", {2, 3}},
            {"data_offsets", {0, 24}} // 2*3*4字节 = 24字节
        };
        header["test_weight_2"] = {
            {"dtype", "F32"},
            {"shape", {1, 4}},
            {"data_offsets", {24, 40}} // 1*4*4字节 = 16字节
        };

        std::string header_str = header.dump();
        uint64_t header_size = header_str.size();

        // Create test weight data
        std::vector<float> weight1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}; // 2x3
        std::vector<float> weight2 = {7.0f, 8.0f, 9.0f, 10.0f};            // 1x4

        // Write to file
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        // Write header length
        file.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));

        // Write header
        file.write(header_str.c_str(), header_size);

        // Write weight data
        file.write(reinterpret_cast<const char *>(weight1.data()), weight1.size() * sizeof(float));
        file.write(reinterpret_cast<const char *>(weight2.data()), weight2.size() * sizeof(float));

        file.close();
        return true;
    }
};

// Test SafeTensors constructor and basic functionality
TEST_F(SafeTensorsTest, BasicFunctionality)
{
    // Create test file
    ASSERT_TRUE(create_test_safetensors_file(test_file_path));

    // Load safetensors file
    SafeTensors st(test_file_path);

    // Test getting header
    json header = st.get_header();
    EXPECT_TRUE(header.contains("test_weight_1"));
    EXPECT_TRUE(header.contains("test_weight_2"));

    // Verify header content for test_weight_1
    const auto &weight1_info = header["test_weight_1"];
    EXPECT_EQ(weight1_info["dtype"], "F32");
    std::vector<int> expected_shape1 = {2, 3};
    EXPECT_EQ(weight1_info["shape"], expected_shape1);
    std::vector<int> expected_offsets1 = {0, 24};
    EXPECT_EQ(weight1_info["data_offsets"], expected_offsets1);

    // Verify header content for test_weight_2
    const auto &weight2_info = header["test_weight_2"];
    EXPECT_EQ(weight2_info["dtype"], "F32");
    std::vector<int> expected_shape2 = {1, 4};
    EXPECT_EQ(weight2_info["shape"], expected_shape2);
    std::vector<int> expected_offsets2 = {24, 40};
    EXPECT_EQ(weight2_info["data_offsets"], expected_offsets2);
}

// Test getting weight data
TEST_F(SafeTensorsTest, GetWeightData)
{
    // Create test file
    ASSERT_TRUE(create_test_safetensors_file(test_file_path));

    // Load safetensors file
    SafeTensors st(test_file_path);

    // Test getting existing weights
    void *weight1_ptr = st.get_weight("test_weight_1");
    void *weight2_ptr = st.get_weight("test_weight_2");

    ASSERT_NE(weight1_ptr, nullptr);
    ASSERT_NE(weight2_ptr, nullptr);

    // Verify weight data
    float *weight1_data = reinterpret_cast<float *>(weight1_ptr);
    float *weight2_data = reinterpret_cast<float *>(weight2_ptr);

    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> expected2 = {7.0f, 8.0f, 9.0f, 10.0f};

    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_FLOAT_EQ(weight1_data[i], expected1[i]);
    }

    for (size_t i = 0; i < expected2.size(); ++i)
    {
        EXPECT_FLOAT_EQ(weight2_data[i], expected2[i]);
    }
}

// Test error handling for non-existent file
TEST_F(SafeTensorsTest, NonExistentFile)
{
    EXPECT_THROW({ SafeTensors st("/nonexistent/file.bin"); }, std::runtime_error);
}

// Test error handling for empty file
TEST_F(SafeTensorsTest, EmptyFile)
{
    const std::string empty_file = "/tmp/empty_test.bin";
    std::ofstream(empty_file).close(); // Create empty file

    EXPECT_THROW({ SafeTensors st(empty_file); }, std::runtime_error);

    std::remove(empty_file.c_str());
}

// Test error handling for file too small to contain header
TEST_F(SafeTensorsTest, FileTooSmall)
{
    const std::string small_file = "/tmp/small_test.bin";
    std::ofstream file(small_file, std::ios::binary);

    // Write only 4 bytes (less than required 8 bytes for header size)
    uint32_t dummy = 0;
    file.write(reinterpret_cast<const char *>(&dummy), sizeof(dummy));
    file.close();

    EXPECT_THROW({ SafeTensors st(small_file); }, std::runtime_error);

    std::remove(small_file.c_str());
}

// Test error handling for invalid JSON header
TEST_F(SafeTensorsTest, InvalidJsonHeader)
{
    const std::string invalid_file = "/tmp/invalid_json_test.bin";
    std::ofstream file(invalid_file, std::ios::binary);

    // Write header size
    uint64_t header_size = 10;
    file.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));

    // Write invalid JSON
    std::string invalid_json = "not json!";
    file.write(invalid_json.c_str(), header_size);

    file.close();

    EXPECT_THROW({ SafeTensors st(invalid_file); }, std::runtime_error);

    std::remove(invalid_file.c_str());
}

TEST(Qwen3Safetensors, Test)
{
    std::string file_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/model.safetensors";
    SafeTensors st(file_path);
    auto header = st.get_header();
    // auto head_str = header.dump(4); // 打印header内容
    // LOG(INFO) << "Header: \n"
    //           << head_str;
}