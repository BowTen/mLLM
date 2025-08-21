#include "kernel/kernel.h"
#include "base/tensor.h"
#include "tokenizer/tokenizer.h"
#include "model/qwen3.h"
#include "base/util.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

class EmbeddingKernelTest : public ::testing::Test
{
protected:
    Tensor cpu_input;
    Tensor cpu_weight;
    Tensor cpu_output;
    Tensor cuda_input;
    Tensor cuda_weight;
    Tensor cuda_output;

    size_t vocab_size;
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> output_shape;

    void SetUp() override
    {
        google::InitGoogleLogging("EmbeddingKernelTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Setting up test environment";

        vocab_size = 8;
        batch_size = 4;
        seq_len = 2;
        hidden_size = 4;
        input_shape = {batch_size, seq_len, 1};
        weight_shape = {vocab_size, hidden_size};
        output_shape = {batch_size, seq_len, hidden_size};

        std::vector<uint32_t> input_data = {4, 1,
                                            6, 3,
                                            4, 2,
                                            1, 7};
        std::vector<float> weight_data = {1, 1, 4, 9,
                                          2, 3, 1, 2,
                                          3, 2, 5, 6,
                                          4, 7, 9, 8,
                                          5, 2, 7, 3,
                                          6, 1, 5, 8,
                                          7, 2, 4, 7,
                                          8, 3, 6, 8};

        LOG(INFO) << "Init Tensors";
        cpu_input = Tensor(input_data.data(), input_shape, true, Device::CPU);
        cpu_weight = Tensor(weight_data.data(), weight_shape, true, Device::CPU);
        cpu_output = Tensor(output_shape, Device::CPU);
        VLOG(DEBUG) << "input size: " << cpu_input.size()
                    << ", weight size: " << cpu_weight.size()
                    << ", output size: " << cpu_output.size();
        cuda_input = cpu_input.clone();
        cuda_input.toDevice(Device::CUDA);
        cuda_weight = cpu_weight.clone();
        cuda_weight.toDevice(Device::CUDA);
        cuda_output = Tensor(output_shape, Device::CUDA);
    }

    void TearDown() override
    {
        // Clean up if needed
        google::ShutdownGoogleLogging();
    }
};

TEST_F(EmbeddingKernelTest, CPUvsCUDA)
{
    LOG(INFO) << "Running CPU vs CUDA test";
    kernel::get_emb_kernel(Device::CUDA)(&cuda_input, &cuda_weight, &cuda_output, hidden_size, nullptr);
    kernel::get_emb_kernel(Device::CPU)(&cpu_input, &cpu_weight, &cpu_output, hidden_size, nullptr);
    cuda_output.toDevice(Device::CPU);

    cout << "CPU Output:" << endl;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            for (size_t k = 0; k < hidden_size; k++)
            {
                cout << *(cpu_output[{i, j, k}]) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    cout << "CUDA Output:" << endl;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            for (size_t k = 0; k < hidden_size; k++)
            {
                cout << *(cuda_output[{i, j, k}]) << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << endl;

    int diff = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            for (size_t k = 0; k < hidden_size; k++)
            {
                diff += *(cpu_output[{i, j, k}]) != *(cuda_output[{i, j, k}]);
            }
        }
    }
    EXPECT_EQ(diff, 0);
}

class TokenizerEmbeddingKernelTest : public ::testing::Test
{
protected:
    tokenizer::BPETokenizer tokenizer = tokenizer::BPETokenizer::from_file("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/tokenizer.json");
    base::json config = base::load_json("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/config.json");
    op::Embedding embedding = op::Embedding(tokenizer.vocab_size(), config["hidden_size"], Device::CPU, nullptr);
    base::SafeTensors safetensors = base::SafeTensors("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/model.safetensors");
    Tensor cpu_input;
    Tensor cpu_weight;
    Tensor cpu_output;
    Tensor cuda_input;
    Tensor cuda_weight;
    Tensor cuda_output;

    size_t vocab_size;
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> output_shape;

    void SetUp() override
    {
        google::InitGoogleLogging("EmbeddingKernelTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Setting up test environment";

        embedding.loadWeight("model.embed_tokens", safetensors);

        vocab_size = tokenizer.vocab_size();
        batch_size = 4;
        seq_len = 2;
        hidden_size = config["hidden_size"];
        input_shape = {batch_size, seq_len, 1};
        weight_shape = {vocab_size, hidden_size};
        output_shape = {batch_size, seq_len, hidden_size};

        std::vector<uint32_t> input_data = {4, 1,
                                            6, 3,
                                            4, 2,
                                            1, 7};

        LOG(INFO) << "Init Tensors";
        cpu_input = Tensor(input_data.data(), input_shape, true, Device::CPU);
        cpu_weight = embedding.getWeight();
        cpu_output = Tensor(output_shape, Device::CPU);
        VLOG(DEBUG) << "input size: " << cpu_input.size()
                    << ", weight size: " << cpu_weight.size()
                    << ", output size: " << cpu_output.size();
        cuda_input = cpu_input.clone();
        cuda_input.toDevice(Device::CUDA);
        cuda_weight = cpu_weight.clone();
        cuda_weight.toDevice(Device::CUDA);
        cuda_output = Tensor(output_shape, Device::CUDA);
    }

    void TearDown() override
    {
        // Clean up if needed
        google::ShutdownGoogleLogging();
    }
};

TEST_F(TokenizerEmbeddingKernelTest, CPUvsCUDAQwen3)
{
    LOG(INFO) << "Running CPU vs CUDA test";
    kernel::get_emb_kernel(Device::CUDA)(&cuda_input, &cuda_weight, &cuda_output, hidden_size, nullptr);
    kernel::get_emb_kernel(Device::CPU)(&cpu_input, &cpu_weight, &cpu_output, hidden_size, nullptr);
    cuda_output.toDevice(Device::CPU);

    auto cuda_shape = cuda_output.shape();
    auto cpu_shape = cpu_output.shape();
    cout << "cpu shape: ";
    for (auto x : cpu_shape)
    {
        cout << x << " ";
    }
    cout << '\n';
    cout << "cuda shape: ";
    for (auto x : cuda_shape)
    {
        cout << x << " ";
    }
    cout << '\n';

    int diff = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            for (size_t k = 0; k < hidden_size; k++)
            {
                diff += *(cpu_output[{i, j, k}]) != *(cuda_output[{i, j, k}]);
            }
        }
    }
    cout << "cuda output: ";
    for (int i = 0; i < 5; i++)
        cout << *cuda_output[i] << ' ';
    cout << "...\n";
    cout << "cpu output: ";
    for (int i = 0; i < 5; i++)
        cout << *cpu_output[i] << ' ';
    cout << "...\n";

    EXPECT_EQ(diff, 0);
}