#include <gtest/gtest.h>
#include "base/util.h"
#include "qwen3_check.hpp"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

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
