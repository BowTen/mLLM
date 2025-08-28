#ifndef MLLM_TOKENIZER_H
#define MLLM_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace mllm
{
    namespace tokenizer
    {

        class BPETokenizer
        {
        public:
            static const uint32_t QWEN3_END_OF_TEXT = 151643;
            static const uint32_t QWEN3_IM_START = 151644;
            static const uint32_t QWEN3_IM_END = 151645;

        private:
            std::vector<std::pair<std::string, uint32_t>> sorted_vocab;
            std::vector<std::string> vocab;
            size_t num_special_tokens;

            BPETokenizer(std::string tokenizer_path);

            static std::string get_unicode(uint32_t c);

            static std::unordered_map<std::string, uint8_t> get_byte_map();

            static std::unordered_map<uint8_t, std::string> get_inv_byte_map();

        public:
            static BPETokenizer from_file(const std::string &tokenizer_path);

            std::vector<uint32_t> encode(const std::string &text, bool special_token = false) const;

            std::string decode(const std::vector<uint32_t> &ids) const;

            std::string decode(uint32_t id) const;

            size_t vocab_size() const { return vocab.size(); }
        };

    } // namespace tokenizer
} // namespace mllm

#endif // MLLM_TOKENIZER_H
