#include "model/qwen3.h"
#include <gtest/gtest.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

int main(int argc, char **argv)
{
    testing::InitGoogleTest();
    google::InitGoogleLogging(argv[0]);
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

    Qwen3Test() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CPU, 0.1f))
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
    string input_text = "你";
    auto ids = tokenizer->encode(input_text);
    cout << input_text << " -> ";
    for (auto id : ids)
        cout << id << ", ";
    cout << endl;

    Tensor ids_tensor({ids.size()}, Device::CPU, false);
    for (size_t i = 0; i < ids.size(); ++i)
        *reinterpret_cast<uint32_t *>(ids_tensor[i]) = ids[i];

    Tensor next_id({1}, Device::CPU, false);

    LOG(INFO) << "Model forward...";
    model.forward(ids_tensor, next_id);
    LOG(INFO) << "Model forward completed.";

    cout << "next id: " << *reinterpret_cast<uint32_t *>(next_id[0]) << endl;
}