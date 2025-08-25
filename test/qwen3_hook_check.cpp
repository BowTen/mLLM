#include "model/qwen3.h"
#include <gtest/gtest.h>
#include <cnpy.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

int main()
{
    testing::InitGoogleTest();
    google::InitGoogleLogging("Qwen3HookCheck");
    FLAGS_logtostderr = true;

    // 运行所有测试
    int result = RUN_ALL_TESTS();

    // 清理 Google Logging
    google::ShutdownGoogleLogging();

    return result;
}

class Qwen3HookCheck : public ::testing::Test
{
protected:
    float check_eps = 1e-3;
    std::string outputs_path = "/home/hznuojai/ai_infra/MiniLLM/scripts/debug_outputs";
    Qwen3 model;

    Qwen3HookCheck() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", base::Device::CPU, 0.5f))
    {
        VLOG(DEBUG) << "Set up Qwen3HookCheck";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3HookCheck";
    }

    static string shape_str(std::vector<size_t> shape)
    {
        string s = "[";
        for (auto dim : shape)
        {
            s += to_string(dim) + ",";
        }
        return s + "]";
    }

    static float relative_diff(float a, float b)
    {
        if (a == b)
            return 0.0f;
        return std::min(std::fabs(a - b) / std::max(std::fabs(a), std::fabs(b)), std::fabs(a - b));
    }

    void check_hook(WLayer *layer)
    {
        auto name = layer->name();
        std::replace(name.begin(), name.end(), '.', '_');
        auto right_output = cnpy::npy_load(outputs_path + "/" + name + ".npy");
        float *right_output_data = right_output.data<float>();

        auto output = layer->getOutput(0);
        output.contiguous();
        float *output_data = output.data();
        size_t output_size = output.size();

        VLOG(DEBUG) << "Check " << layer->name() << " output shape: " << shape_str(output.shape());
        for (size_t i = 0; i < output_size; i++)
        {
            float diff = relative_diff(output_data[i], right_output_data[i]);
            CHECK_LT(diff, check_eps);
        }
    }
};

TEST_F(Qwen3HookCheck, Demo)
{
    LOG(INFO) << "Run Demo";
    auto tokenizer = model.get_tokenizer();
    string input_text = "The weather is really ";
    auto ids = tokenizer->encode(input_text);
    cout << input_text << " -> ";
    for (auto id : ids)
        cout << id << ", ";
    cout << endl;

    Tensor ids_tensor = Tensor::from_vector(ids, {ids.size(), 1}, Device::CPU, false);
    Tensor next_id({1, 1}, Device::CPU, false);

    LOG(INFO) << "Setting Hook...";
    model.register_hooks([this](WLayer *layer)
                         { this->check_hook(layer); });

    LOG(INFO) << "Model forward...";
    model.forward(ids_tensor, next_id);
    LOG(INFO) << "Model forward completed.";

    size_t next_id_value = *reinterpret_cast<uint32_t *>(next_id[0]);
    cout << "next id: " << next_id_value << endl;

    cout << "decode token: " << tokenizer->decode(next_id_value) << endl;
}