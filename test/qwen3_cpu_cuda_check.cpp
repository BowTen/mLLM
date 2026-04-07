#include "qwen3_check.hpp"
#include "base/util.h"
#include <algorithm>

using namespace std;
using namespace mllm;
using namespace mllm::base;
using namespace mllm::model;

namespace
{
    std::vector<float> tensor_to_fp32_vector(Tensor tensor)
    {
        tensor.toDevice(Device::CPU);
        std::vector<float> values(tensor.logic_size(), 0.0f);
        materialize_float_storage(tensor.raw_data(), tensor.dtype(), values.data(), DType::FP32, values.size());
        return values;
    }

    void expect_valid_probability_distribution(const std::vector<float> &values,
                                               float sum_tolerance,
                                               const std::string &label)
    {
        float sum = 0.0f;
        bool has_positive_mass = false;
        for (size_t i = 0; i < values.size(); ++i)
        {
            ASSERT_TRUE(std::isfinite(values[i])) << label << " index " << i;
            EXPECT_GE(values[i], 0.0f) << label << " index " << i;
            EXPECT_LE(values[i], 1.0f) << label << " index " << i;
            sum += values[i];
            has_positive_mass = has_positive_mass || values[i] > 0.0f;
        }
        EXPECT_TRUE(has_positive_mass) << label;
        EXPECT_NEAR(sum, 1.0f, sum_tolerance) << label;
    }

    void expect_probability_reference_drift_bounded(const std::vector<float> &reference,
                                                    const std::vector<float> &candidate,
                                                    float max_abs_tolerance,
                                                    const std::string &label)
    {
        ASSERT_EQ(reference.size(), candidate.size()) << label;
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < reference.size(); ++i)
        {
            max_abs_diff = std::max(max_abs_diff, std::abs(reference[i] - candidate[i]));
        }
        EXPECT_LE(max_abs_diff, max_abs_tolerance) << label;
    }

    size_t floating_weight_storage_bytes(Qwen3 &model)
    {
        size_t total_bytes = 0;
        for (auto *layer : model.weighted_layers())
        {
            if (is_floating_point_dtype(layer->weight().dtype()))
            {
                total_bytes += layer->weight().buffer()->size();
            }
        }
        return total_bytes;
    }
} // namespace

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
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

TEST(Qwen3Bf16RegressionTest, Bf16RunsWithNormalizedProbabilitiesAndReducesWeightStorage)
{
    const std::string model_path = resolve_qwen3_model_path();
    const std::vector<uint32_t> token_ids = {1234u, 5678u, 4356u, 1246u, 9870u};

    {
        Qwen3 fp32_cpu = Qwen3::from_pretrained(model_path, Device::CPU, 5.0f, 20, 0.95f, 0.0f, DType::FP32);
        Qwen3 bf16_cpu = Qwen3::from_pretrained(model_path, Device::CPU, 5.0f, 20, 0.95f, 0.0f, DType::BF16);

        Tensor input_fp32 = Tensor::from_vector(token_ids, {token_ids.size(), 1}, Device::CPU, false, nullptr);
        Tensor input_bf16 = Tensor::from_vector(token_ids, {token_ids.size(), 1}, Device::CPU, false, nullptr);
        Tensor next_fp32({1, 1}, Device::CPU, false, nullptr, DType::U32);
        Tensor next_bf16({1, 1}, Device::CPU, false, nullptr, DType::U32);

        fp32_cpu.forward(input_fp32, next_fp32);
        bf16_cpu.forward(input_bf16, next_bf16);

        const std::vector<float> fp32_values = tensor_to_fp32_vector(fp32_cpu.final_probability);
        const std::vector<float> bf16_values = tensor_to_fp32_vector(bf16_cpu.final_probability);
        expect_valid_probability_distribution(fp32_values, 1e-3f, "cpu_fp32_final_probability");
        expect_valid_probability_distribution(bf16_values, 1e-2f, "cpu_bf16_final_probability");
        expect_probability_reference_drift_bounded(fp32_values, bf16_values, 0.5f, "cpu_bf16_reference_drift");

        const size_t fp32_bytes = floating_weight_storage_bytes(fp32_cpu);
        const size_t bf16_bytes = floating_weight_storage_bytes(bf16_cpu);
        LOG(INFO) << "CPU floating weight storage bytes fp32=" << fp32_bytes << " bf16=" << bf16_bytes;
        EXPECT_LT(bf16_bytes, fp32_bytes);
    }

    {
        Qwen3 fp32_cuda = Qwen3::from_pretrained(model_path, Device::CUDA, 5.0f, 20, 0.95f, 0.0f, DType::FP32);
        Qwen3 bf16_cuda = Qwen3::from_pretrained(model_path, Device::CUDA, 5.0f, 20, 0.95f, 0.0f, DType::BF16);

        Tensor input_fp32 = Tensor::from_vector(token_ids, {token_ids.size(), 1}, Device::CUDA, false, fp32_cuda.stream());
        Tensor input_bf16 = Tensor::from_vector(token_ids, {token_ids.size(), 1}, Device::CUDA, false, bf16_cuda.stream());
        Tensor next_fp32({1, 1}, Device::CUDA, false, fp32_cuda.stream(), DType::U32);
        Tensor next_bf16({1, 1}, Device::CUDA, false, bf16_cuda.stream(), DType::U32);

        fp32_cuda.forward(input_fp32, next_fp32);
        bf16_cuda.forward(input_bf16, next_bf16);

        const std::vector<float> fp32_values = tensor_to_fp32_vector(fp32_cuda.final_probability);
        const std::vector<float> bf16_values = tensor_to_fp32_vector(bf16_cuda.final_probability);
        expect_valid_probability_distribution(fp32_values, 1e-3f, "cuda_fp32_final_probability");
        expect_valid_probability_distribution(bf16_values, 1e-2f, "cuda_bf16_final_probability");
        expect_probability_reference_drift_bounded(fp32_values, bf16_values, 0.5f, "cuda_bf16_reference_drift");

        const size_t fp32_bytes = floating_weight_storage_bytes(fp32_cuda);
        const size_t bf16_bytes = floating_weight_storage_bytes(bf16_cuda);
        LOG(INFO) << "CUDA floating weight storage bytes fp32=" << fp32_bytes << " bf16=" << bf16_bytes;
        EXPECT_LT(bf16_bytes, fp32_bytes);
    }
}
