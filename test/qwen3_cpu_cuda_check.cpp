#include "qwen3_check.hpp"

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

int main()
{
    testing::InitGoogleTest();
    google::InitGoogleLogging("Qwen3CpuCudaCheckTest");
    FLAGS_logtostderr = true;

    // 运行所有测试
    int result = RUN_ALL_TESTS();

    // 清理 Google Logging
    google::ShutdownGoogleLogging();

    return result;
}

class Qwen3CpuCudaCheckTest : public ::testing::Test
{
protected:
    float check_eps = 1e-6f;
    Qwen3Check qwen3_check;

    Qwen3CpuCudaCheckTest()
        : qwen3_check("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", 5.0f, 1e-2f)
    {
        VLOG(DEBUG) << "Set up Qwen3CpuCudaCheckTest";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3CpuCudaCheckTest";
    }
};

TEST_F(Qwen3CpuCudaCheckTest, Demo)
{
    LOG(INFO) << "Run Qwen3 Check";
    qwen3_check.forward({1234});
}