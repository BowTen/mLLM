#include <gtest/gtest.h>
#include "base/util.h"
#include "qwen3_check.hpp"
#include <ctime>
#include <random>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

namespace mllm::base
{
    extern std::mt19937 global_mt;
}

namespace
{
    std::vector<float> tensor_to_fp32_vector(Tensor tensor)
    {
        tensor.toDevice(Device::CPU);
        std::vector<float> values(tensor.logic_size(), 0.0f);
        materialize_float_storage(tensor.raw_data(), tensor.dtype(), values.data(), DType::FP32, values.size());
        return values;
    }

    void expect_valid_probabilities(const Tensor &probabilities)
    {
        const std::vector<float> values = tensor_to_fp32_vector(probabilities);
        float sum = 0.0f;
        bool has_positive_mass = false;
        for (float value : values)
        {
            ASSERT_TRUE(std::isfinite(value));
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, 1.0f);
            sum += value;
            has_positive_mass = has_positive_mass || value > 0.0f;
        }
        EXPECT_TRUE(has_positive_mass);
        EXPECT_NEAR(sum, 1.0f, 1e-2f);
    }
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("Qwen3Test");
    FLAGS_logtostderr = true;

    // 运行所有测试
    int result = RUN_ALL_TESTS();

    // 清理 Google Logging
    google::ShutdownGoogleLogging();

    return result;
}

class Qwen3Test : public ::testing::Test
{
protected:
    std::string model_path;
    Qwen3 model;

    Qwen3Test()
        : model_path(resolve_qwen3_model_path()),
          model(Qwen3::from_pretrained(model_path, base::Device::CPU, 0.3f, 20, 0.95f, 0.0f, DType::FP32))
    {
        VLOG(DEBUG) << "Set up Qwen3Test";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3Test";
    }
};

TEST(Qwen3ConstructionTest, DefaultsFloatingPointStorageToBF16)
{
    Qwen3 model = Qwen3::from_pretrained(resolve_qwen3_model_path(), Device::CPU, 0.3f);

    EXPECT_EQ(model.inference_dtype(), DType::BF16);
    EXPECT_EQ(model.embed_tokens.weight().dtype(), DType::BF16);
    EXPECT_EQ(model.norm.weight().dtype(), DType::BF16);
    EXPECT_EQ(model.lm_head.weight().dtype(), DType::BF16);
    EXPECT_EQ(model.rotary_embedding.inv_freq.dtype(), DType::FP32);
    EXPECT_EQ(model.temperature_scaling.dtype(), DType::FP32);
    EXPECT_EQ(model.final_probability.dtype(), DType::FP32);
    ASSERT_FALSE(model.layers.empty());
    EXPECT_EQ(model.layers.front().self_attn.q_proj.weight().dtype(), DType::BF16);
    EXPECT_EQ(model.layers.front().mlp.gate_proj.weight().dtype(), DType::BF16);
}

TEST(Qwen3ConstructionTest, AllowsExplicitFP32InferenceOverride)
{
    Qwen3 model = Qwen3::from_pretrained(resolve_qwen3_model_path(), Device::CPU, 0.3f, 20, 0.95f, 0.0f, DType::FP32);

    EXPECT_EQ(model.inference_dtype(), DType::FP32);
    EXPECT_EQ(model.embed_tokens.weight().dtype(), DType::FP32);
    EXPECT_EQ(model.norm.weight().dtype(), DType::FP32);
    EXPECT_EQ(model.lm_head.weight().dtype(), DType::FP32);
    EXPECT_EQ(model.rotary_embedding.inv_freq.dtype(), DType::FP32);
    EXPECT_EQ(model.temperature_scaling.dtype(), DType::FP32);
    EXPECT_EQ(model.final_probability.dtype(), DType::FP32);
}

TEST(Qwen3ConstructionTest, DefaultBF16ForwardPreservesDynamicTensorDTypes)
{
    Qwen3 model = Qwen3::from_pretrained(resolve_qwen3_model_path(), Device::CPU, 0.3f);
    Tensor input_id = Tensor::from_vector(std::vector<uint32_t>{151644u}, {1, 1}, Device::CPU, false, nullptr);
    Tensor next_id({1, 1}, Device::CPU, false, nullptr, DType::U32);

    model.forward(input_id, next_id);

    EXPECT_EQ(model.hidden_state.dtype(), DType::BF16);
    EXPECT_EQ(model.cos.dtype(), DType::BF16);
    EXPECT_EQ(model.sin.dtype(), DType::BF16);
    EXPECT_EQ(model.final_probability.dtype(), DType::FP32);
    ASSERT_FALSE(model.layers.empty());
    EXPECT_EQ(model.layers.front().self_attn.k_cache.dtype(), DType::BF16);
    EXPECT_EQ(model.layers.front().self_attn.v_cache.dtype(), DType::BF16);
    EXPECT_EQ(next_id.dtype(), DType::U32);
}

TEST(Qwen3ConstructionTest, LoadsSharded8BOnCpu)
{
    std::string model_path;
    try
    {
        model_path = resolve_qwen3_8b_model_path();
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << e.what();
    }
    Qwen3 model = Qwen3::from_pretrained(model_path, Device::CPU, 0.3f);

    EXPECT_EQ(model.device(), Device::CPU);
    EXPECT_EQ(model.inference_dtype(), DType::BF16);
    EXPECT_FALSE(model.layers.empty());
    EXPECT_EQ(model.embed_tokens.weight().dtype(), DType::BF16);
    EXPECT_EQ(model.lm_head.weight().dtype(), DType::BF16);
}

TEST(SamplingKernelRegressionTest, TopKFilteringRenormalizesProbabilityMass)
{
    Tensor probabilities = Tensor::from_vector(std::vector<float>{0.40f, 0.35f, 0.25f}, {1, 3}, Device::CPU, false, nullptr);
    Tensor sampled_token({1, 1}, Device::CPU, false, nullptr, DType::U32);

    global_mt.seed(1);
    kernel::sampling_kernel(&probabilities, &sampled_token, 1, 1.0f, 0.0f);

    EXPECT_EQ(sampled_token.data<uint32_t>()[0], 0u);
    global_mt.seed(static_cast<uint32_t>(time(nullptr)));
}

TEST(SamplingKernelRegressionTest, SamplingRespectsValidTokenizerVocabPrefix)
{
    Tensor probabilities = Tensor::from_vector(std::vector<float>{0.05f, 0.05f, 0.05f, 0.40f, 0.45f}, {1, 5}, Device::CPU, false, nullptr);
    Tensor sampled_token({1, 1}, Device::CPU, false, nullptr, DType::U32);

    global_mt.seed(1);
    kernel::sampling_kernel(&probabilities, &sampled_token, 5, 1.0f, 0.0f, 3);

    EXPECT_LT(sampled_token.data<uint32_t>()[0], 3u);
    global_mt.seed(static_cast<uint32_t>(time(nullptr)));
}

TEST(Qwen3LargeModelCpuSmokeTest, Sharded8BCompletesShortDecodeLoop)
{
    std::string model_path;
    try
    {
        model_path = resolve_qwen3_8b_model_path();
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << e.what();
    }
    Qwen3 model = Qwen3::from_pretrained(model_path, Device::CPU, 0.3f);
    auto tokenizer = model.get_tokenizer();
    std::string input_text = "人类: 请用一句话介绍你自己。\n助手:";
    std::vector<uint32_t> ids = tokenizer->encode(input_text);
    ASSERT_FALSE(ids.empty());

    Tensor input_id = Tensor::from_vector(ids, {ids.size(), 1}, Device::CPU, false, nullptr);
    Tensor next_id({1, 1}, Device::CPU, false, nullptr, DType::U32);

    std::string generated_text;
    for (int step = 0; step < 2; ++step)
    {
        model.forward(input_id, next_id);
        expect_valid_probabilities(model.final_probability);

        const uint32_t token = next_id.data<uint32_t>()[0];
        generated_text += tokenizer->decode(token);
        input_id = next_id.clone();
    }

    EXPECT_FALSE(generated_text.empty());
}

TEST_F(Qwen3Test, Demo)
{
    LOG(INFO) << "Run Demo";
    auto tokenizer = model.get_tokenizer();
    string input_text = "下面是一段人工智能助手与人类的对话：\n人类:你是谁？\n助手:你好呀！我是Qwen3！\n人类:你会做什么？\n";
    auto ids = tokenizer->encode(input_text);
    cout << input_text << " -> ";
    for (auto id : ids)
        cout << id << ", ";
    cout << endl;

    Tensor input_id = Tensor::from_vector(ids, {ids.size(), 1}, Device::CPU, false, nullptr);
    Tensor next_id({1, 1}, Device::CPU, false, nullptr, DType::U32);

    for (int i = 0; i < 100; i++)
    {
        LOG(INFO) << "Model forward round " << i << "...";
        model.forward(input_id, next_id);
        LOG(INFO) << "Model forward completed.";

        size_t next_id_value = next_id.data<uint32_t>()[0];
        cout << "next id: " << next_id_value << endl;
        cout << "decode token: " << tokenizer->decode(next_id_value) << endl;
        input_text += tokenizer->decode(next_id_value);

        input_id = next_id.clone();
    }

    cout << "Final text: \n";
    cout << input_text << endl;
}

// class Qwen3TestCuda : public ::testing::Test
// {
// protected:
//     Qwen3 model;

//     Qwen3TestCuda() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CUDA, 0.5f))
//     {
//         VLOG(DEBUG) << "Set up Qwen3TestCuda";
//     }

//     void TearDown() override
//     {
//         VLOG(DEBUG) << "Tearing down Qwen3TestCuda";
//     }
// };

// TEST_F(Qwen3TestCuda, Demo)
// {
//     LOG(INFO) << "Run Demo";
//     auto tokenizer = model.get_tokenizer();
//     string input_text = "下面是一段人工智能助手与人类的对话：\n人类:你是谁？\n助手:你好呀！我是Qwen3！\n人类:你会做什么？\n";
//     auto ids = tokenizer->encode(input_text);
//     cout << input_text << " -> ";
//     for (auto id : ids)
//         cout << id << ", ";
//     cout << endl;

//     Tensor input_id = Tensor::from_vector(ids, {ids.size(), 1}, Device::CUDA, false, model.stream());
//     Tensor next_id({1, 1}, Device::CUDA, false, model.stream());

//     for (int i = 0; i < 10; i++)
//     {
//         LOG(INFO) << "Model forward...";
//         // CHECK_CUDA_ERR(cudaDeviceSynchronize());
//         model.forward(input_id, next_id);
//         // CHECK_CUDA_ERR(cudaDeviceSynchronize());
//         LOG(INFO) << "Model forward completed.";
//         next_id.toDevice(Device::CPU);
//         size_t next_id_value = *reinterpret_cast<uint32_t *>(next_id[0]);
//         cout << "next id: " << next_id_value << endl;
//         cout << "decode token: " << tokenizer->decode(next_id_value) << endl;
//         input_text += tokenizer->decode(next_id_value);

//         next_id.toDevice(Device::CUDA);
//         input_id = next_id.clone();
//     }

//     cout << "Final text: \n"
//          << input_text << endl;
// }
