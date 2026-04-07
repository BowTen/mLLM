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
    float check_eps = 1e-3f;
    Qwen3Check qwen3_check;

    Qwen3CpuCudaCheckTest()
        : qwen3_check(resolve_qwen3_model_path(), 5.0f, check_eps)
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
    qwen3_check.forward_model_and_check({1234, 5678, 4356, 1246, 9870});
    qwen3_check.forward_model_and_check({910});
}
