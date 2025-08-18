#include "model/qwen3.h"
#include <gtest/gtest.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace mllm;

base::Tensor forward_to_embedding(mllm::model::Qwen3 *qwen3, std::string text)
{
    auto tokens = qwen3->get_tokenizer()->encode(text);
    std::vector<size_t> shape = {tokens.size(), 1};
    base::Tensor input(shape, base::Device::CPU);
    uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_data[i] = tokens[i];
    }
    input.toDevice(qwen3->device());

    auto config = qwen3->config();
    auto hidden_size = config["hidden_size"].get<size_t>();
    std::vector<size_t> out_shape = {tokens.size(), hidden_size};
    base::Tensor output(out_shape, qwen3->device());

    qwen3->get_embed_tokens()->setInput(0, input);
    qwen3->get_embed_tokens()->setOutput(0, output);
    qwen3->get_embed_tokens()->forward();

    return output;
}

base::Tensor forward_to_norm(mllm::model::Qwen3 *qwen3, std::string text)
{
    auto tokens = qwen3->get_tokenizer()->encode(text);
    std::vector<size_t> shape = {tokens.size(), 1};
    base::Tensor input(shape, base::Device::CPU);
    uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_data[i] = tokens[i];
    }
    input.toDevice(qwen3->device());

    auto config = qwen3->config();
    auto hidden_size = config["hidden_size"].get<size_t>();
    std::vector<size_t> out_shape = {tokens.size(), hidden_size};
    base::Tensor output(out_shape, qwen3->device());

    qwen3->get_embed_tokens()->setInput(0, input);
    qwen3->get_embed_tokens()->setOutput(0, output);
    qwen3->get_embed_tokens()->forward();

    base::Tensor norm_output(out_shape, qwen3->device());
    qwen3->get_norm()->setInput(0, output);
    qwen3->get_norm()->setOutput(0, norm_output);
    qwen3->get_norm()->forward();

    return norm_output;
}

class Qwen3CPU : public ::testing::Test
{
protected:
    model::Qwen3 qwen3;

    Qwen3CPU()
        : qwen3(model::Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CPU))
    {
        LOG(INFO) << "Initialized Qwen3 model for CPU tests";
    }

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
class Qwen3CUDA : public ::testing::Test
{
protected:
    model::Qwen3 qwen3;

    Qwen3CUDA()
        : qwen3(model::Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CUDA))
    {
        LOG(INFO) << "Initialized Qwen3 model for CUDA tests";
    }

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

TEST_F(Qwen3CPU, TestEmbeddingCPU)
{
    using mllm::base::Tensor;

    // 测试输入文本
    std::string input_text = "Hello, world!";

    // 执行前向传播
    Tensor output;
    EXPECT_NO_THROW({
        output = forward_to_embedding(&qwen3, input_text);
    });

    auto model_config = qwen3.config();

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
        for (int j = 0; j < 5; j++)
        {
            output_str += std::to_string(output_data[i * shape[1] + j]) + " ";
        }
        output_str += "...";
        LOG(INFO) << output_str;
    }
}

TEST_F(Qwen3CPU, TestNormCPU)
{
    using mllm::base::Tensor;

    // 测试输入文本
    std::string input_text = "Hello, world!";

    // 执行前向传播
    Tensor output;
    EXPECT_NO_THROW({
        output = forward_to_norm(&qwen3, input_text);
    });

    auto model_config = qwen3.config();

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
        for (int j = 0; j < 5; j++)
        {
            output_str += std::to_string(output_data[i * shape[1] + j]) + " ";
        }
        output_str += "...";
        LOG(INFO) << output_str;
    }
}

TEST_F(Qwen3CUDA, TestEmbeddingCUDA)
{
    using mllm::base::Tensor;

    // 测试输入文本
    std::string input_text = "Hello, world!";

    // 执行前向传播
    Tensor output;
    EXPECT_NO_THROW({
        output = forward_to_embedding(&qwen3, input_text);
        output.toDevice(mllm::base::Device::CPU); // 转移 output到CPU以便验证
    });

    auto model_config = qwen3.config();

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
        for (int j = 0; j < 5; j++)
        {
            output_str += std::to_string(output_data[i * shape[1] + j]) + " ";
        }
        output_str += "...";
        LOG(INFO) << output_str;
    }
}

TEST_F(Qwen3CUDA, TestNormCUDA)
{
    using mllm::base::Tensor;

    // 测试输入文本
    std::string input_text = "Hello, world!";

    // 执行前向传播
    Tensor output;
    EXPECT_NO_THROW({
        output = forward_to_norm(&qwen3, input_text);
        output.toDevice(mllm::base::Device::CPU); // 转移 output到CPU以便验证
    });

    auto model_config = qwen3.config();

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
        for (int j = 0; j < 5; j++)
        {
            output_str += std::to_string(output_data[i * shape[1] + j]) + " ";
        }
        output_str += "...";
        LOG(INFO) << output_str;
    }
}