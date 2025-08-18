#include "model/qwen3.h"
#include <gtest/gtest.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

class Qwen3Test : public ::testing::Test
{
protected:
    void SetUp() override
    { // Initialize any necessary resources or configurations
        google::InitGoogleLogging("Qwen3Test");
        FLAGS_logtostderr = true; // Log to stderr for visibility in tests
        LOG(INFO) << "Setting up Qwen3Test";
    }

    void TearDown() override
    {
        // Clean up resources if needed
        google::ShutdownGoogleLogging();
    }
};

TEST_F(Qwen3Test, test)
{
    using mllm::base::Tensor;

    std::string model_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B";
    mllm::model::Qwen3 model = mllm::model::Qwen3::from_pretrained(model_path);

    // 测试输入文本
    std::string input_text = "Hello, world!";

    // 执行前向传播
    Tensor output;
    EXPECT_NO_THROW({
        output = model.forward_test(input_text);
    });

    auto model_config = model.config();

    // 检查输出形状是否正确
    auto shape = output.shape();

    EXPECT_EQ(shape.size(), 2); // 应该是二维张量

    EXPECT_EQ(shape[0], 4); // "Hello, world!" 分词后长度为4
    LOG(INFO) << "Verified sequence length: " << shape[0] << std::endl;

    EXPECT_EQ(shape[1], model_config["hidden_size"]); // 隐藏层大小应与模型配置匹配

    auto output_data = output.data();
    for (size_t i = 0; i < shape[0]; i++)
    {
        std::string output_str = "Output[" + std::to_string(i) + "]: ";
        for (int j = 0; j < 10; j++)
        {
            output_str += std::to_string(output_data[i * shape[1] + j]) + " ";
        }
        output_str += "...";
        LOG(INFO) << output_str;
    }
}