#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include "tokenizer/tokenizer.h"

using namespace mllm::tokenizer;

// 全局变量存储tokenizer路径
std::string g_tokenizer_path = "/home/hznuojai/ai_infra/MiniLLM/resources/Qwen/Qwen3-0.6B/tokenizer.json";

class TokenizerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 如果全局路径存在且文件存在，使用全局路径
        if (!g_tokenizer_path.empty() && std::filesystem::exists(g_tokenizer_path))
        {
            test_tokenizer_path = g_tokenizer_path;
            use_custom_tokenizer = false;
        }
        else
        {
            // 否则创建测试用tokenizer文件
            test_tokenizer_path = "test_tokenizer.json";
            use_custom_tokenizer = true;
            createTestTokenizerFile();
        }
    }

    void TearDown() override
    {
        // 只清理自定义创建的测试文件
        if (use_custom_tokenizer && std::filesystem::exists(test_tokenizer_path))
        {
            std::filesystem::remove(test_tokenizer_path);
        }
    }

    void createTestTokenizerFile()
    {
        std::string tokenizer_content = R"({
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                    "ab": 3,
                    "bc": 4,
                    "abc": 5,
                    " ": 6,
                    "hello": 7,
                    "world": 8,
                    "!": 9,
                    "test": 10
                }
            },
            "merges": [
                ["a", "b"],
                ["b", "c"],
                ["ab", "c"]
            ],
            "added_tokens": [
                {
                    "id": 11,
                    "content": "<unk>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }
            ]
        })";

        std::ofstream file(test_tokenizer_path);
        file << tokenizer_content;
        file.close();
    }

    std::string test_tokenizer_path;
    bool use_custom_tokenizer;
};

// 测试tokenizer文件加载
TEST_F(TokenizerTest, LoadTokenizer)
{
    EXPECT_NO_THROW({
        auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);
    });
}

// 测试无效文件路径
TEST_F(TokenizerTest, InvalidTokenizerPath)
{
    EXPECT_THROW({ auto tokenizer = BPETokenizer::from_file("non_existent_file.json"); }, std::runtime_error);
}

// 测试基本编码功能
TEST_F(TokenizerTest, BasicEncoding)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    // 测试简单单词编码
    std::vector<uint32_t> encoded = tokenizer.encode("test");
    EXPECT_FALSE(encoded.empty());

    // 测试包含空格的文本
    std::vector<uint32_t> encoded_with_space = tokenizer.encode(" test");
    EXPECT_FALSE(encoded_with_space.empty());
}

// 测试编码-解码往返
TEST_F(TokenizerTest, EncodeDecodeRoundTrip)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    std::string original_text = "test";
    auto encoded = tokenizer.encode(original_text);
    std::string decoded_text = tokenizer.decode(encoded);

    EXPECT_EQ(original_text, decoded_text);
}

// 测试空字符串处理
TEST_F(TokenizerTest, EmptyString)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    EXPECT_THROW({ auto encoded = tokenizer.encode(""); }, std::runtime_error);
}

// 测试无效token ID解码
TEST_F(TokenizerTest, InvalidTokenId)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    EXPECT_THROW({
        tokenizer.decode(2000000); // 不存在的token ID
    },
                 std::out_of_range);
}

// 测试多字符处理
TEST_F(TokenizerTest, MultiCharacterProcessing)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    // 测试包含多个已知token的文本
    std::string text = "hello world!";
    auto encoded = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(encoded);

    EXPECT_EQ(text, decoded);
}

TEST_F(TokenizerTest, NoneSpecialTokens)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    std::string end_token = "<|im_end|>， <|endoftext|>";
    auto ids = tokenizer.encode(end_token, false);

    std::cout << "txt: " << end_token << std::endl;
    std::cout << "ids: ";
    for (const auto &id : ids)
    {
        std::cout << id << " ";
    }
    std::cout << std::endl;

    auto decode = tokenizer.decode(ids);
    std::cout << "decode: " << decode << std::endl;

    std::vector<uint32_t> expected_ids({27, 91, 318, 6213, 91, 29, 3837, 82639, 8691, 69, 1318, 91, 29});
    EXPECT_EQ(ids, expected_ids);
    EXPECT_EQ(end_token, decode);
}
TEST_F(TokenizerTest, SpecialTokens)
{
    auto tokenizer = BPETokenizer::from_file(test_tokenizer_path);

    std::string end_token = "<|im_end|>， <|endoftext|>";
    auto ids = tokenizer.encode(end_token, true);

    std::cout << "txt: " << end_token << std::endl;
    std::cout << "ids: ";
    for (const auto &id : ids)
    {
        std::cout << id << " ";
    }
    std::cout << std::endl;
    auto decode = tokenizer.decode(ids);
    std::cout << "decode: " << decode << std::endl;

    std::vector<uint32_t> expected_ids({151645, 3837, 220, 151643});
    EXPECT_EQ(ids, expected_ids);
    EXPECT_EQ(end_token, decode);
}

int main(int argc, char **argv)
{
    // 处理命令行参数
    if (argc > 1)
    {
        g_tokenizer_path = argv[1];
        std::cout << "Using tokenizer file: " << g_tokenizer_path << std::endl;

        // 检查文件是否存在
        if (!std::filesystem::exists(g_tokenizer_path))
        {
            std::cerr << "Error: Tokenizer file does not exist: " << g_tokenizer_path << std::endl;
            std::cerr << "Falling back to default test tokenizer..." << std::endl;
            g_tokenizer_path.clear();
        }
    }
    else
    {
        std::cout << "No tokenizer file specified, using default test tokenizer." << std::endl;
        std::cout << "Usage: " << argv[0] << " [tokenizer.json]" << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
