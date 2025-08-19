#include "kernel/kernel.h"
#include "base/tensor.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace mllm;
using namespace mllm::base;

class EmbeddingKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize test parameters
        vocab_size = 100;
        hidden_size = 64;
        seq_length = 5;
    }

    void TearDown() override
    {
        // Clean up if needed
    }

    size_t vocab_size;
    size_t hidden_size;
    size_t seq_length;
};

TEST_F(EmbeddingKernelTest, BasicEmbeddingCPU0)
{
    Tensor inputs({seq_length});
    Tensor weights({vocab_size, hidden_size});
    Tensor outputs({seq_length, hidden_size});
    uint32_t *input_data = reinterpret_cast<uint32_t *>(inputs.data());
    float *weight_data = weights.data();
    float *output_data = outputs.data();

    // Initialize input token indices
    for (size_t i = 0; i < seq_length; ++i)
    {
        input_data[i] = (i * 2) % vocab_size; // Safe indices
    }

    // Initialize weight matrix with test values
    for (size_t i = 0; i < vocab_size; ++i)
    {
        for (size_t j = 0; j < hidden_size; ++j)
        {
            weight_data[i * hidden_size + j] = static_cast<float>(i + j * 0.1);
        }
    }

    // Get CPU kernel and execute
    auto kernel = mllm::kernel::get_emb_kernel(mllm::base::Device::CPU);
    ASSERT_NE(kernel, nullptr);

    // Execute embedding kernel
    EXPECT_NO_THROW({
        kernel(&inputs, &weights, &outputs, vocab_size, hidden_size, nullptr);
    });

    // Verify output: each output row should match the corresponding weight row
    for (size_t i = 0; i < seq_length; ++i)
    {
        uint32_t token_idx = input_data[i];
        for (size_t j = 0; j < hidden_size; ++j)
        {
            float expected = static_cast<float>(token_idx + j * 0.1);
            float actual = output_data[i * hidden_size + j];
            EXPECT_FLOAT_EQ(expected, actual)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}

TEST_F(EmbeddingKernelTest, BasicEmbeddingCPU)
{
    // Create input tensor (token indices)
    std::vector<size_t> input_shape = {seq_length};
    base::Tensor input(input_shape, base::Device::CPU);

    // Fill input with some token indices
    uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
    for (size_t i = 0; i < seq_length; ++i)
    {
        input_data[i] = i % vocab_size; // Safe indices
    }

    // Create weight tensor (embedding matrix)
    std::vector<size_t> weight_shape = {vocab_size, hidden_size};
    base::Tensor weight(weight_shape, base::Device::CPU);

    // Fill weight with some test values
    float *weight_data = weight.data();
    for (size_t i = 0; i < vocab_size; ++i)
    {
        for (size_t j = 0; j < hidden_size; ++j)
        {
            weight_data[i * hidden_size + j] = static_cast<float>(i + j * 0.1);
        }
    }

    // Create output tensor
    std::vector<size_t> output_shape = {seq_length, hidden_size};
    base::Tensor output(output_shape, base::Device::CPU);

    // Get CPU kernel
    auto kernel = kernel::get_emb_kernel(base::Device::CPU);
    ASSERT_NE(kernel, nullptr);

    // Execute kernel
    EXPECT_NO_THROW({
        kernel(&input, &weight, &output, vocab_size, hidden_size, nullptr);
    });

    // Verify output
    float *output_data = output.data();
    for (size_t i = 0; i < seq_length; ++i)
    {
        uint32_t token_idx = input_data[i];
        for (size_t j = 0; j < hidden_size; ++j)
        {
            float expected = static_cast<float>(token_idx + j * 0.1);
            float actual = output_data[i * hidden_size + j];
            EXPECT_FLOAT_EQ(expected, actual)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}

TEST_F(EmbeddingKernelTest, OutOfRangeIndex)
{
    // Create input tensor with out-of-range index
    std::vector<size_t> input_shape = {1};
    base::Tensor input(input_shape, base::Device::CPU);

    uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
    input_data[0] = vocab_size + 10; // Out of range

    // Create weight tensor
    std::vector<size_t> weight_shape = {vocab_size, hidden_size};
    base::Tensor weight(weight_shape, base::Device::CPU);

    // Create output tensor
    std::vector<size_t> output_shape = {1, hidden_size};
    base::Tensor output(output_shape, base::Device::CPU);

    // Get CPU kernel
    auto kernel = kernel::get_emb_kernel(base::Device::CPU);

    // Expect exception for out-of-range index
    EXPECT_THROW({ kernel(&input, &weight, &output, vocab_size, hidden_size, nullptr); }, std::out_of_range);
}

TEST_F(EmbeddingKernelTest, EmptyInput)
{
    // Create empty input tensor
    std::vector<size_t> input_shape = {0};
    base::Tensor input(input_shape, base::Device::CPU);

    // Create weight tensor
    std::vector<size_t> weight_shape = {vocab_size, hidden_size};
    base::Tensor weight(weight_shape, base::Device::CPU);

    // Create empty output tensor
    std::vector<size_t> output_shape = {0, hidden_size};
    base::Tensor output(output_shape, base::Device::CPU);

    // Get CPU kernel
    auto kernel = kernel::get_emb_kernel(base::Device::CPU);

    // Should handle empty input gracefully
    EXPECT_NO_THROW({
        kernel(&input, &weight, &output, vocab_size, hidden_size, nullptr);
    });
}

TEST_F(EmbeddingKernelTest, LargeEmbedding)
{
    // Test with larger dimensions
    size_t large_vocab_size = 1000;
    size_t large_hidden_size = 256;
    size_t large_seq_length = 20;

    // Create input tensor
    std::vector<size_t> input_shape = {large_seq_length};
    base::Tensor input(input_shape, base::Device::CPU);

    uint32_t *input_data = reinterpret_cast<uint32_t *>(input.data());
    for (size_t i = 0; i < large_seq_length; ++i)
    {
        input_data[i] = i * 50 % large_vocab_size; // Various indices
    }

    // Create weight tensor
    std::vector<size_t> weight_shape = {large_vocab_size, large_hidden_size};
    base::Tensor weight(weight_shape, base::Device::CPU);

    float *weight_data = weight.data();
    for (size_t i = 0; i < large_vocab_size * large_hidden_size; ++i)
    {
        weight_data[i] = static_cast<float>(std::sin(i * 0.01)); // Some pattern
    }

    // Create output tensor
    std::vector<size_t> output_shape = {large_seq_length, large_hidden_size};
    base::Tensor output(output_shape, base::Device::CPU);

    // Get CPU kernel
    auto kernel = kernel::get_emb_kernel(base::Device::CPU);

    // Execute kernel
    EXPECT_NO_THROW({
        kernel(&input, &weight, &output, large_vocab_size, large_hidden_size, nullptr);
    });

    // Verify a few random positions
    float *output_data = output.data();
    for (size_t i = 0; i < std::min(large_seq_length, size_t(5)); ++i)
    {
        uint32_t token_idx = input_data[i];
        for (size_t j = 0; j < std::min(large_hidden_size, size_t(5)); ++j)
        {
            float expected = weight_data[token_idx * large_hidden_size + j];
            float actual = output_data[i * large_hidden_size + j];
            EXPECT_FLOAT_EQ(expected, actual);
        }
    }
}