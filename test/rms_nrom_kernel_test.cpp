#include "kernel/kernel.h"
#include "base/tensor.h"
#include "model/qwen3.h"
#include "base/util.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using namespace std;
using namespace mllm;
using namespace mllm::base;

class Qwen3RMSNormKernelTest : public ::testing::Test
{
protected:
    base::json config = base::load_json("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/config.json");
    op::RMSNorm norm = op::RMSNorm(config["hidden_size"], config["rms_norm_eps"], Device::CPU, nullptr);
    base::SafeTensors safetensors = base::SafeTensors("/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/model.safetensors");
    Tensor cpu_input;
    Tensor cpu_weight;
    Tensor cpu_output;
    Tensor cuda_input;
    Tensor cuda_weight;
    Tensor cuda_output;

    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    std::vector<size_t> input_shape;
    std::vector<size_t> weight_shape;
    std::vector<size_t> output_shape;

    std::mt19937 rnd = std::mt19937(std::random_device{}());
    std::uniform_real_distribution<> urd = std::uniform_real_distribution<>(0.0, 1.0);

    float check_eps = 1e-3;

    void SetUp() override
    {
        google::InitGoogleLogging("EmbeddingKernelTest");
        FLAGS_logtostderr = true;
        VLOG(DEBUG) << "Setting up test environment";

        norm.loadWeight("model.norm", safetensors, false);

        batch_size = 4;
        seq_len = 4;
        hidden_size = config["hidden_size"];
        input_shape = {batch_size, seq_len, hidden_size};
        weight_shape = {1, hidden_size};
        output_shape = {batch_size, seq_len, hidden_size};

        LOG(INFO) << "Init Tensors";
        cpu_input = Tensor(input_shape);
        for (size_t i = 0; i < cpu_input.size(); i++)
        {
            *cpu_input[i] = urd(rnd);
        }

        cpu_weight = norm.getWeight();
        cpu_output = Tensor(output_shape);
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

TEST_F(Qwen3RMSNormKernelTest, CPUvsCUDAQwen3)
{
    LOG(INFO) << "Running CPU vs CUDA test";
    kernel::get_rmsnorm_kernel(Device::CUDA)(&cuda_input, &cuda_weight, &cuda_output, hidden_size, nullptr);
    kernel::get_rmsnorm_kernel(Device::CPU)(&cpu_input, &cpu_weight, &cpu_output, hidden_size, nullptr);
    cuda_input.toDevice(Device::CPU);
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
                diff += std::fabs(*(cpu_output[{i, j, k}]) - *(cuda_output[{i, j, k}])) > check_eps;
            }
        }
    }
    EXPECT_EQ(diff, 0);
    for (size_t i = 0; i < batch_size; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            cout << "batch: " << i << ", token: " << j << ":\n";
            cout << "cpu input: ";
            for (size_t k = 0; k < 5; k++)
                cout << *cpu_input[{i, j, k}] << " ";
            cout << "...\n";
            cout << "cpu output: ";
            for (size_t k = 0; k < 5; k++)
                cout << *cpu_output[{i, j, k}] << " ";
            cout << "...\n";

            cout << "cuda input: ";
            for (size_t k = 0; k < 5; k++)
                cout << *cuda_input[{i, j, k}] << " ";
            cout << "...\n";
            cout << "cuda output: ";
            for (size_t k = 0; k < 5; k++)
                cout << *cuda_output[{i, j, k}] << " ";
            cout << "...\n";

            if (diff == 0)
                break;
        }
        if (diff == 0)
            break;
    }
}

class RMSNormKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test parameters
        seq_length = 4;
        hidden_size = 8;
        eps = 1e-6f;

        // Setup random number generator for reproducible tests
        std::random_device rd;
        gen.seed(42); // Fixed seed for reproducible tests
        dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
    }

    void TearDown() override
    {
        // Clean up if needed
    }

    // Helper function to fill tensor with random data
    void fillTensorRandom(Tensor &tensor)
    {
        float *data = tensor.data();
        size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = dist(gen);
        }
    }

    // Helper function to fill tensor with specific values
    void fillTensorValue(Tensor &tensor, float value)
    {
        float *data = tensor.data();
        size_t size = tensor.size();
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = value;
        }
    }

    // Helper function to check if two tensors are approximately equal
    bool tensorsApproxEqual(const Tensor &a, const Tensor &b, float tolerance = 1e-5f)
    {
        if (a.shape() != b.shape())
            return false;

        const float *data_a = const_cast<Tensor &>(a).data();
        const float *data_b = const_cast<Tensor &>(b).data();
        size_t size = a.size();

        for (size_t i = 0; i < size; ++i)
        {
            if (std::abs(data_a[i] - data_b[i]) > tolerance)
            {
                std::cout << "Mismatch at index " << i << ": "
                          << data_a[i] << " vs " << data_b[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    // Manual RMSNorm implementation for reference
    void referenceRMSNorm(const Tensor &input, const Tensor &weight, Tensor &output, float eps)
    {
        auto shape = input.shape();
        size_t seq_len = shape[0];
        size_t hidden_dim = shape[1];

        const float *input_data = const_cast<Tensor &>(input).data();
        const float *weight_data = const_cast<Tensor &>(weight).data();
        float *output_data = output.data();

        for (size_t i = 0; i < seq_len; ++i)
        {
            // Calculate RMS
            float sum_sq = 0.0f;
            for (size_t j = 0; j < hidden_dim; ++j)
            {
                float val = input_data[i * hidden_dim + j];
                sum_sq += val * val;
            }
            float rms = std::sqrt(sum_sq / hidden_dim + eps);

            // Apply normalization and scaling
            for (size_t j = 0; j < hidden_dim; ++j)
            {
                output_data[i * hidden_dim + j] =
                    (input_data[i * hidden_dim + j] / rms) * weight_data[j];
            }
        }
    }

    size_t seq_length;
    size_t hidden_size;
    float eps;
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist;
};

TEST_F(RMSNormKernelTest, BasicRMSNormCPU)
{
    // Create tensors
    Tensor input({seq_length, hidden_size}, Device::CPU);
    Tensor weight({hidden_size}, Device::CPU);
    Tensor output({seq_length, hidden_size}, Device::CPU);
    Tensor reference_output({seq_length, hidden_size}, Device::CPU);

    // Fill input with random data
    fillTensorRandom(input);

    // Fill weight with ones for simple test
    fillTensorValue(weight, 1.0f);

    // Get CPU kernel and run
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output, eps, nullptr);

    // Calculate reference result
    referenceRMSNorm(input, weight, reference_output, eps);

    // Compare results
    EXPECT_TRUE(tensorsApproxEqual(output, reference_output, 1e-5f))
        << "CPU RMSNorm output doesn't match reference implementation";
}

TEST_F(RMSNormKernelTest, RMSNormCPUWithRandomWeights)
{
    // Create tensors
    Tensor input({seq_length, hidden_size}, Device::CPU);
    Tensor weight({hidden_size}, Device::CPU);
    Tensor output({seq_length, hidden_size}, Device::CPU);
    Tensor reference_output({seq_length, hidden_size}, Device::CPU);

    // Fill tensors with random data
    fillTensorRandom(input);
    fillTensorRandom(weight);

    // Get CPU kernel and run
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output, eps, nullptr);

    // Calculate reference result
    referenceRMSNorm(input, weight, reference_output, eps);

    // Compare results
    EXPECT_TRUE(tensorsApproxEqual(output, reference_output, 1e-5f))
        << "CPU RMSNorm with random weights doesn't match reference implementation";
}

TEST_F(RMSNormKernelTest, RMSNormCPUZeroInput)
{
    // Create tensors
    Tensor input({seq_length, hidden_size}, Device::CPU);
    Tensor weight({hidden_size}, Device::CPU);
    Tensor output({seq_length, hidden_size}, Device::CPU);

    // Fill input with zeros
    fillTensorValue(input, 0.0f);

    // Fill weight with ones
    fillTensorValue(weight, 1.0f);

    // Get CPU kernel and run
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output, eps, nullptr);

    // Output should be zero (0 / sqrt(eps) * weight = 0)
    float *output_data = output.data();
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_NEAR(output_data[i], 0.0f, 1e-6f)
            << "Zero input should produce zero output at index " << i;
    }
}

TEST_F(RMSNormKernelTest, BasicRMSNormCUDA)
{
    // Create tensors on CPU first
    Tensor input({seq_length, hidden_size}, Device::CPU);
    Tensor weight({hidden_size}, Device::CPU);
    Tensor output({seq_length, hidden_size}, Device::CPU);
    Tensor reference_output({seq_length, hidden_size}, Device::CPU);

    // Fill CPU tensors with random data
    fillTensorRandom(input);
    fillTensorValue(weight, 1.0f);

    // Calculate reference result on CPU first
    referenceRMSNorm(input, weight, reference_output, eps);

    // Move tensors to GPU
    input.toDevice(Device::CUDA);
    weight.toDevice(Device::CUDA);
    output.toDevice(Device::CUDA);

    // Get CUDA kernel and run
    auto cuda_kernel = kernel::get_rmsnorm_kernel(Device::CUDA);
    cuda_kernel(&input, &weight, &output, eps, nullptr);

    // Move result back to CPU for comparison
    output.toDevice(Device::CPU);

    // Compare results
    EXPECT_TRUE(tensorsApproxEqual(output, reference_output, 1e-4f))
        << "CUDA RMSNorm output doesn't match reference implementation";
}

TEST_F(RMSNormKernelTest, CPUvsCUDAConsistency)
{
    // Create tensors on CPU
    Tensor input({seq_length, hidden_size}, Device::CPU);
    Tensor weight({hidden_size}, Device::CPU);
    Tensor output_cpu({seq_length, hidden_size}, Device::CPU);
    Tensor output_cuda({seq_length, hidden_size}, Device::CPU);

    // Fill tensors with random data
    fillTensorRandom(input);
    fillTensorRandom(weight);

    // Run CPU kernel
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output_cpu, eps, nullptr);

    // Create copies for CUDA test
    Tensor input_cuda({seq_length, hidden_size}, Device::CPU);
    Tensor weight_cuda({hidden_size}, Device::CPU);

    // Copy data to CUDA tensors
    std::memcpy(input_cuda.data(), input.data(), input.size() * sizeof(float));
    std::memcpy(weight_cuda.data(), weight.data(), weight.size() * sizeof(float));

    // Move to GPU
    input_cuda.toDevice(Device::CUDA);
    weight_cuda.toDevice(Device::CUDA);
    output_cuda.toDevice(Device::CUDA);

    // Run CUDA kernel
    auto cuda_kernel = kernel::get_rmsnorm_kernel(Device::CUDA);
    cuda_kernel(&input_cuda, &weight_cuda, &output_cuda, eps, nullptr);

    // Move CUDA result back to CPU for comparison
    output_cuda.toDevice(Device::CPU);

    // Compare CPU and CUDA results
    EXPECT_TRUE(tensorsApproxEqual(output_cpu, output_cuda, 1e-4f))
        << "CPU and CUDA RMSNorm results are not consistent";
}

TEST_F(RMSNormKernelTest, LargerTensorCUDATest)
{
    // Test CUDA with larger tensors
    size_t large_seq = 32;
    size_t large_hidden = 256;

    // Create tensors on CPU first
    Tensor input({large_seq, large_hidden}, Device::CPU);
    Tensor weight({large_hidden}, Device::CPU);
    Tensor output_cpu({large_seq, large_hidden}, Device::CPU);
    Tensor output_cuda({large_seq, large_hidden}, Device::CPU);

    // Fill tensors with random data
    fillTensorRandom(input);
    fillTensorRandom(weight);

    // Run CPU kernel for reference
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output_cpu, eps, nullptr);

    // Create copies for CUDA test
    Tensor input_cuda({large_seq, large_hidden}, Device::CPU);
    Tensor weight_cuda({large_hidden}, Device::CPU);

    // Copy data
    std::memcpy(input_cuda.data(), input.data(), input.size() * sizeof(float));
    std::memcpy(weight_cuda.data(), weight.data(), weight.size() * sizeof(float));

    // Move to GPU
    input_cuda.toDevice(Device::CUDA);
    weight_cuda.toDevice(Device::CUDA);
    output_cuda.toDevice(Device::CUDA);

    // Run CUDA kernel
    auto cuda_kernel = kernel::get_rmsnorm_kernel(Device::CUDA);
    cuda_kernel(&input_cuda, &weight_cuda, &output_cuda, eps, nullptr);

    // Move result back to CPU
    output_cuda.toDevice(Device::CPU);

    // Compare results
    EXPECT_TRUE(tensorsApproxEqual(output_cpu, output_cuda, 1e-4f))
        << "Large tensor CUDA RMSNorm doesn't match CPU implementation";
}

TEST_F(RMSNormKernelTest, CUDAWithDifferentEpsValues)
{
    // Test CUDA with different epsilon values
    std::vector<float> eps_values = {1e-8f, 1e-6f, 1e-4f, 1e-2f};

    for (float test_eps : eps_values)
    {
        // Create tensors on CPU
        Tensor input({seq_length, hidden_size}, Device::CPU);
        Tensor weight({hidden_size}, Device::CPU);
        Tensor output_cpu({seq_length, hidden_size}, Device::CPU);
        Tensor output_cuda({seq_length, hidden_size}, Device::CPU);

        // Fill tensors with random data
        fillTensorRandom(input);
        fillTensorRandom(weight);

        // Run CPU kernel
        auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
        cpu_kernel(&input, &weight, &output_cpu, test_eps, nullptr);

        // Create copies for CUDA
        Tensor input_cuda({seq_length, hidden_size}, Device::CPU);
        Tensor weight_cuda({hidden_size}, Device::CPU);

        std::memcpy(input_cuda.data(), input.data(), input.size() * sizeof(float));
        std::memcpy(weight_cuda.data(), weight.data(), weight.size() * sizeof(float));

        // Move to GPU
        input_cuda.toDevice(Device::CUDA);
        weight_cuda.toDevice(Device::CUDA);
        output_cuda.toDevice(Device::CUDA);

        // Run CUDA kernel
        auto cuda_kernel = kernel::get_rmsnorm_kernel(Device::CUDA);
        cuda_kernel(&input_cuda, &weight_cuda, &output_cuda, test_eps, nullptr);

        // Move result back
        output_cuda.toDevice(Device::CPU);

        // Compare results
        EXPECT_TRUE(tensorsApproxEqual(output_cpu, output_cuda, 1e-4f))
            << "CUDA RMSNorm with eps=" << test_eps << " doesn't match CPU implementation";
    }
}

TEST_F(RMSNormKernelTest, LargerTensorTest)
{
    // Test with larger tensors
    size_t large_seq = 16;
    size_t large_hidden = 128;

    Tensor input({large_seq, large_hidden}, Device::CPU);
    Tensor weight({large_hidden}, Device::CPU);
    Tensor output({large_seq, large_hidden}, Device::CPU);
    Tensor reference_output({large_seq, large_hidden}, Device::CPU);

    // Fill tensors with random data
    fillTensorRandom(input);
    fillTensorRandom(weight);

    // Get CPU kernel and run
    auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
    cpu_kernel(&input, &weight, &output, eps, nullptr);

    // Calculate reference result
    referenceRMSNorm(input, weight, reference_output, eps);

    // Compare results
    EXPECT_TRUE(tensorsApproxEqual(output, reference_output, 1e-5f))
        << "Large tensor RMSNorm doesn't match reference implementation";
}

TEST_F(RMSNormKernelTest, DifferentEpsValues)
{
    // Test with different epsilon values
    std::vector<float> eps_values = {1e-8f, 1e-6f, 1e-4f, 1e-2f};

    for (float test_eps : eps_values)
    {
        Tensor input({seq_length, hidden_size}, Device::CPU);
        Tensor weight({hidden_size}, Device::CPU);
        Tensor output({seq_length, hidden_size}, Device::CPU);
        Tensor reference_output({seq_length, hidden_size}, Device::CPU);

        // Fill tensors with random data
        fillTensorRandom(input);
        fillTensorRandom(weight);

        // Get CPU kernel and run
        auto cpu_kernel = kernel::get_rmsnorm_kernel(Device::CPU);
        cpu_kernel(&input, &weight, &output, test_eps, nullptr);

        // Calculate reference result
        referenceRMSNorm(input, weight, reference_output, test_eps);

        // Compare results
        EXPECT_TRUE(tensorsApproxEqual(output, reference_output, 1e-5f))
            << "RMSNorm with eps=" << test_eps << " doesn't match reference implementation";
    }
}