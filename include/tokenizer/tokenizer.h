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
        private:
            std::vector<std::pair<std::string, uint32_t>> sorted_vocab;
            std::vector<std::string> vocab;

            BPETokenizer(std::string tokenizer_path);

            static std::string get_unicode(uint32_t c);

            static std::unordered_map<std::string, uint8_t> get_byte_map();

            static std::unordered_map<uint8_t, std::string> get_inv_byte_map();

        public:
            static BPETokenizer from_file(const std::string &tokenizer_path);

            std::vector<uint32_t> encode(const std::string &text) const;

            std::string decode(const std::vector<uint32_t> &ids) const;

            std::string decode(uint32_t id) const;
        };

    } // namespace tokenizer
} // namespace mllm

#endif // MLLM_TOKENIZER_H
