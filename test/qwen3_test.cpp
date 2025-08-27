#include "model/qwen3.h"
#include <gtest/gtest.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

int main()
{
    testing::InitGoogleTest();
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
    Qwen3 model;

    Qwen3Test() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CPU, 1.0f))
    {
        VLOG(DEBUG) << "Set up Qwen3Test";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3Test";
    }
};

TEST_F(Qwen3Test, Demo)
{
    LOG(INFO) << "Run Demo";
    auto tokenizer = model.get_tokenizer();
    string input_text = "The weather is really 2";
    auto ids = tokenizer->encode(input_text);
    cout << input_text << " -> ";
    for (auto id : ids)
        cout << id << ", ";
    cout << endl;

    Tensor input_id = Tensor::from_vector(ids, {ids.size(), 1}, Device::CPU, false);
    Tensor next_id({1, 1}, Device::CPU, false);

    for (int i = 0; i < 10; i++)
    {
        LOG(INFO) << "Model forward round " << i << "...";
        model.forward(input_id, next_id);
        LOG(INFO) << "Model forward completed.";

        size_t next_id_value = *reinterpret_cast<uint32_t *>(next_id[0]);
        cout << "next id: " << next_id_value << endl;
        cout << "decode token: " << tokenizer->decode(next_id_value) << endl;
        input_text += tokenizer->decode(next_id_value);

        input_id = next_id.clone();
    }

    cout << "Final text: \n";
    cout << input_text << endl;
}
// CPU Result:
// Top 10 tokens:
// Token ID: 1023, token_str: lock, Probability: 2.75253e-05
// Token ID: 5600, token_str: dom, Probability: 2.73505e-05
// Token ID: 344, token_str: iv, Probability: 2.73494e-05
// Token ID: 6591, token_str: ini, Probability: 2.7133e-05
// Token ID: 4648, token_str: has, Probability: 2.70131e-05
// Token ID: 1466, token_str: ah, Probability: 2.68349e-05
// Token ID: 24794, token_str: ammed, Probability: 2.66178e-05
// Token ID: 1524, token_str: iver, Probability: 2.63897e-05
// Token ID: 484, token_str: ind, Probability: 2.63745e-05
// Token ID: 49746, token_str: mess, Probability: 2.63377e-05

// class Qwen3TestCuda : public ::testing::Test
// {
// protected:
//     Qwen3 model;

//     Qwen3TestCuda() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CUDA, 1.0f))
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
//     string input_text = "The weather is really ";
//     auto ids = tokenizer->encode(input_text);
//     cout << input_text << " -> ";
//     for (auto id : ids)
//         cout << id << ", ";
//     cout << endl;

//     Tensor input_id = Tensor::from_vector(ids, {ids.size(), 1}, Device::CUDA, false);
//     Tensor next_id({1, 1}, Device::CUDA, false);

//     for (int i = 0; i < 10; i++)
//     {
//         LOG(INFO) << "Model forward...";
//         model.forward(input_id, next_id);
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

// Cuda Result:
// Top 10 tokens:
// Token ID: 1023, token_str: lock, Probability: 2.75266e-05
// Token ID: 5600, token_str: dom, Probability: 2.73517e-05
// Token ID: 344, token_str: iv, Probability: 2.73507e-05
// Token ID: 6591, token_str: ini, Probability: 2.71343e-05
// Token ID: 4648, token_str: has, Probability: 2.70143e-05
// Token ID: 1466, token_str: ah, Probability: 2.68362e-05
// Token ID: 24794, token_str: ammed, Probability: 2.6619e-05
// Token ID: 1524, token_str: iver, Probability: 2.63909e-05
// Token ID: 484, token_str: ind, Probability: 2.63757e-05
// Token ID: 49746, token_str: mess, Probability: 2.6339e-05