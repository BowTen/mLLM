#include "model/qwen3.h"
#include <gtest/gtest.h>
#include "base/util.h"
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
    google::InitGoogleLogging("Qwen3IncrementalCheck");
    FLAGS_logtostderr = true;

    // 运行所有测试
    int result = RUN_ALL_TESTS();

    // 清理 Google Logging
    google::ShutdownGoogleLogging();

    return result;
}

class Qwen3IncrementalCheck : public ::testing::Test
{
protected:
    base::Device check_device = base::Device::CPU;
    float check_eps = 1e-3;
    std::string outputs_path = "/home/hznuojai/ai_infra/MiniLLM/scripts/debug_outputs";
    Qwen3 model;

    // 用于存储第一步和第二步的hook检查状态
    bool is_step2 = false;

    Qwen3IncrementalCheck() : model(Qwen3::from_pretrained("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B", check_device, 1.0f))
    {
        VLOG(DEBUG) << "Set up Qwen3IncrementalCheck";
    }

    void TearDown() override
    {
        VLOG(DEBUG) << "Tearing down Qwen3IncrementalCheck";
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

    void check_hook_step1(WLayer *layer)
    {
        auto name = layer->name();
        if (name == "lm_head")
            return;

        std::replace(name.begin(), name.end(), '.', '_');
        string filename = outputs_path + "/step1/" + name + ".npy";

        // 检查文件是否存在
        ifstream file(filename);
        if (!file.good())
        {
            LOG(WARNING) << "Reference file not found: " << filename;
            return;
        }

        auto right_output = cnpy::npy_load(filename);
        float *right_output_data = right_output.data<float>();

        auto output = layer->getOutput(0);
        output.contiguous();
        output.toDevice(Device::CPU);
        // CHECK_CUDA_ERR(cudaDeviceSynchronize());

        float *output_data = output.data();
        size_t output_size = output.size();

        VLOG(DEBUG) << "[STEP1] Check " << layer->name() << " output shape: " << shape_str(output.shape());

        // 检查元素总数是否匹配
        size_t expected_size = 1;
        for (auto dim : right_output.shape)
        {
            expected_size *= dim;
        }

        if (output_size != expected_size)
        {
            LOG(ERROR) << "[STEP1] Size mismatch for " << layer->name()
                       << ": expected total elements " << expected_size
                       << ", got " << output_size;
            CHECK(false);
        }

        // 检查从后往前的形状维度是否匹配（允许前面的维度合并）
        auto output_shape = output.shape();
        vector<size_t> ref_shape(right_output.shape.begin(), right_output.shape.end());

        bool shape_compatible = true;
        if (output_shape.size() <= ref_shape.size())
        {
            // 从最后一个维度开始比较
            int output_idx = output_shape.size() - 1;
            int ref_idx = ref_shape.size() - 1;

            while (output_idx >= 0 && ref_idx >= 0)
            {
                if (output_shape[output_idx] != ref_shape[ref_idx])
                {
                    shape_compatible = false;
                    break;
                }
                output_idx--;
                ref_idx--;
            }
        }
        else
        {
            shape_compatible = false;
        }

        if (!shape_compatible)
        {
            LOG(INFO) << "[STEP1] Shape difference for " << layer->name()
                      << ": expected " << shape_str(ref_shape)
                      << ", got " << shape_str(output_shape)
                      << " (but element count matches, proceeding...)";
        }

        for (size_t i = 0; i < output_size; i++)
        {
            float diff = relative_diff(output_data[i], right_output_data[i]);
            if (std::isnan(diff) || std::isinf(diff) || diff >= check_eps)
            {
                LOG(ERROR) << "[STEP1] Layer " << layer->name() << " at index " << i
                           << ": expected " << right_output_data[i]
                           << ", got " << output_data[i]
                           << ", diff " << diff;
                CHECK(false);
            }
        }

        output.toDevice(model.device());
        LOG(INFO) << "[STEP1] Layer " << layer->name() << " check passed";
    }

    void check_hook_step2(WLayer *layer)
    {
        auto name = layer->name();
        if (name == "lm_head")
            return;

        std::replace(name.begin(), name.end(), '.', '_');
        string filename = outputs_path + "/step2/" + name + ".npy";

        // 检查文件是否存在
        ifstream file(filename);
        if (!file.good())
        {
            LOG(WARNING) << "Reference file not found: " << filename;
            return;
        }

        auto right_output = cnpy::npy_load(filename);
        float *right_output_data = right_output.data<float>();

        auto output = layer->getOutput(0);
        output.contiguous();
        output.toDevice(Device::CPU);
        // CHECK_CUDA_ERR(cudaDeviceSynchronize());

        float *output_data = output.data();
        size_t output_size = output.logic_size();

        VLOG(DEBUG) << "[STEP2] Check " << layer->name() << " output shape: " << shape_str(output.shape());

        // 检查元素总数是否匹配
        size_t expected_size = 1;
        for (auto dim : right_output.shape)
        {
            expected_size *= dim;
        }

        if (output_size != expected_size)
        {
            LOG(ERROR) << "[STEP2] Size mismatch for " << layer->name()
                       << ": expected total elements " << expected_size
                       << ", got " << output_size;
            CHECK(false);
        }

        // 检查从后往前的形状维度是否匹配（允许前面的维度合并）
        auto output_shape = output.shape();
        vector<size_t> ref_shape(right_output.shape.begin(), right_output.shape.end());

        bool shape_compatible = true;
        if (output_shape.size() <= ref_shape.size())
        {
            // 从最后一个维度开始比较
            int output_idx = output_shape.size() - 1;
            int ref_idx = ref_shape.size() - 1;

            while (output_idx >= 0 && ref_idx >= 0)
            {
                if (output_shape[output_idx] != ref_shape[ref_idx])
                {
                    shape_compatible = false;
                    break;
                }
                output_idx--;
                ref_idx--;
            }
        }
        else
        {
            shape_compatible = false;
        }

        if (!shape_compatible)
        {
            LOG(INFO) << "[STEP2] Shape difference for " << layer->name()
                      << ": expected " << shape_str(ref_shape)
                      << ", got " << shape_str(output_shape)
                      << " (but element count matches, proceeding...)";
        }

        for (size_t i = 0; i < output_size; i++)
        {
            float diff = relative_diff(output_data[i], right_output_data[i]);
            if (std::isnan(diff) || std::isinf(diff) || diff >= check_eps)
            {
                LOG(ERROR) << "[STEP2] Layer " << layer->name() << " at index " << i
                           << ": expected " << right_output_data[i]
                           << ", got " << output_data[i]
                           << ", diff " << diff;
                CHECK(false);
            }
        }

        output.toDevice(model.device());
        LOG(INFO) << "[STEP2] Layer " << layer->name() << " check passed";
    }
};

TEST_F(Qwen3IncrementalCheck, IncrementalInference)
{
    LOG(INFO) << "Starting Incremental Inference Test";
    auto tokenizer = model.get_tokenizer();

    // 第一步：处理 "The weather is really "
    string first_input_text = "The weather is really ";
    auto first_ids = tokenizer->encode(first_input_text);
    cout << "Step 1 - Input: " << first_input_text << " -> ";
    for (auto id : first_ids)
        cout << id << ", ";
    cout << endl;

    Tensor first_ids_tensor = Tensor::from_vector(first_ids, {first_ids.size(), 1}, check_device, false, model.stream());
    Tensor first_next_id({1, 1}, check_device, false, model.stream());

    LOG(INFO) << "Step 1 - Setting Hook for first forward pass...";
    is_step2 = false;
    model.register_hooks([this](WLayer *layer)
                         { this->check_hook_step1(layer); });

    LOG(INFO) << "Step 1 - Model forward...";
    model.forward(first_ids_tensor, first_next_id);
    LOG(INFO) << "Step 1 - Model forward completed.";

    first_next_id.toDevice(Device::CPU);
    size_t first_next_id_value = *reinterpret_cast<uint32_t *>(first_next_id[0]);
    cout << "Step 1 - next id: " << first_next_id_value << endl;
    cout << "Step 1 - decode token: " << tokenizer->decode(first_next_id_value) << endl;

    // 清除第一步的hooks
    model.clear_hooks();

    // 第二步：处理 "2"（单个token的增量推理）
    string second_input_text = "2";
    auto second_ids = tokenizer->encode(second_input_text);
    cout << "\nStep 2 - Input: " << second_input_text << " -> ";
    for (auto id : second_ids)
        cout << id << ", ";
    cout << endl;

    // 注意：对于增量推理，我们只输入新的token
    Tensor second_ids_tensor = Tensor::from_vector(second_ids, {second_ids.size(), 1}, check_device, false, model.stream());
    Tensor second_next_id({1, 1}, check_device, false, model.stream());

    LOG(INFO) << "Step 2 - Setting Hook for incremental forward pass...";
    is_step2 = true;
    model.register_hooks([this](WLayer *layer)
                         { this->check_hook_step2(layer); });

    LOG(INFO) << "Step 2 - Model incremental forward...";
    model.forward(second_ids_tensor, second_next_id);
    LOG(INFO) << "Step 2 - Model incremental forward completed.";

    second_next_id.toDevice(Device::CPU);
    size_t second_next_id_value = *reinterpret_cast<uint32_t *>(second_next_id[0]);
    cout << "Step 2 - next id: " << second_next_id_value << endl;
    cout << "Step 2 - decode token: " << tokenizer->decode(second_next_id_value) << endl;

    // 验证是否与参考输出匹配
    // 这里可以加载step2_final_logits.npy进行最终验证
    string logits_filename = outputs_path + "/step2/final_logits.npy";
    ifstream logits_file(logits_filename);
    if (logits_file.good())
    {
        auto ref_logits = cnpy::npy_load(logits_filename);
        float *ref_logits_data = ref_logits.data<float>();

        // 找到概率最大的token
        float max_prob = ref_logits_data[0];
        size_t max_idx = 0;
        for (size_t i = 1; i < ref_logits.shape[ref_logits.shape.size() - 1]; i++)
        {
            if (ref_logits_data[i] > max_prob)
            {
                max_prob = ref_logits_data[i];
                max_idx = i;
            }
        }

        cout << "Reference predicted token ID: " << max_idx << endl;
        cout << "Reference predicted token: " << tokenizer->decode(max_idx) << endl;

        // 检查预测是否一致
        EXPECT_EQ(second_next_id_value, max_idx) << "Predicted token mismatch!";
    }

    LOG(INFO) << "Incremental Inference Test completed successfully!";
}
