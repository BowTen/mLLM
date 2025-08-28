#include "tokenizer/tokenizer.h"
#include "base/tensor.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "base/json.hpp"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

using json = nlohmann::json;

namespace mllm
{
    namespace tokenizer
    {
        // Define static constants
        const uint32_t BPETokenizer::QWEN3_END_OF_TEXT;
        const uint32_t BPETokenizer::QWEN3_IM_START;
        const uint32_t BPETokenizer::QWEN3_IM_END;
        const uint32_t BPETokenizer::QWEN3_ENTER;
        const uint32_t BPETokenizer::QWEN3_USER;
        const uint32_t BPETokenizer::QWEN3_ASSISTANT;
        const uint32_t BPETokenizer::QWEN3_THINK;
        const uint32_t BPETokenizer::QWEN3_END_THINK;

        BPETokenizer::BPETokenizer(std::string tokenizer_path)
        {
            VLOG(TRACE) << "Loading tokenizer from: " << tokenizer_path;
            std::ifstream file(tokenizer_path);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open tokenizer file: " +
                                         tokenizer_path);
            }
            json tokenizer_json = json::parse(file);

            if (tokenizer_json["model"]["type"] != "BPE")
            {
                throw std::runtime_error("Tokenizer model type is not BPE.");
            }

            std::unordered_map<std::string, uint32_t> token_frequencies;
            uint32_t rank = UINT32_MAX;
            for (auto &merge : tokenizer_json["merges"])
            {
                std::string token0 = merge[0];
                std::string token1 = merge[1];
                std::string token = token0 + token1;
                token_frequencies[token] = rank--;
            }

            // 将特殊token放在最前面
            for (auto &added_token : tokenizer_json["added_tokens"])
            {
                std::string token = added_token["content"];
                uint32_t id = added_token["id"];
                sorted_vocab.emplace_back(token, id);
            }
            num_special_tokens = sorted_vocab.size();
            for (auto &[token, id] : tokenizer_json["model"]["vocab"].items())
            {
                sorted_vocab.emplace_back(token, id);
            }
            auto byte_map = get_byte_map();
            for (auto &[token, id] : sorted_vocab)
            {
                for (auto &[code, byte] : byte_map)
                {
                    while (true)
                    {
                        auto pos = token.find(code);
                        if (pos == std::string::npos)
                            break;
                        token.replace(pos, code.size(), std::string(1, byte));
                    }
                }
            }
            std::sort(sorted_vocab.begin() + num_special_tokens, sorted_vocab.end(),
                      [&](const auto &a, const auto &b)
                      {
                          if (a.first.size() == b.first.size())
                          {
                              return token_frequencies[a.first] >
                                     token_frequencies[b.first];
                          }
                          return a.first.size() > b.first.size();
                      });

            vocab.resize(sorted_vocab.size());
            for (auto &[token, id] : sorted_vocab)
            {
                vocab[id] = token;
            }
            VLOG(TRACE) << "Successfully loaded tokenizer with vocab size: " << vocab.size();
        }

        std::string BPETokenizer::get_unicode(uint32_t c)
        {
            std::string res;
            if (c < 0x80)
            {
                res += static_cast<char>(c);
            }
            else if (c < 0x800)
            {
                res += static_cast<char>(0xC0 | (c >> 6));
                res += static_cast<char>(0x80 | (c & 0x3F));
            }
            else if (c < 0x10000)
            {
                res += static_cast<char>(0xE0 | (c >> 12));
                res += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                res += static_cast<char>(0x80 | (c & 0x3F));
            }
            else if (c < 0x110000)
            {
                res += static_cast<char>(0xF0 | (c >> 18));
                res += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
                res += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                res += static_cast<char>(0x80 | (c & 0x3F));
            }
            return res;
        }

        std::unordered_map<std::string, uint8_t> BPETokenizer::get_byte_map()
        {
            std::unordered_map<std::string, uint8_t> byte_map;
            std::vector<uint8_t> bs;
            for (uint8_t c = '!'; c <= '~'; c++)
            {
                bs.push_back(c);
            }
            for (uint32_t c = 161; c <= 172; c++)
            {
                bs.push_back(c);
                byte_map[get_unicode(c)] = c;
            }
            for (uint32_t c = 174; c <= 255; c++)
            {
                bs.push_back(c);
                byte_map[get_unicode(c)] = c;
            }

            int n = 0;
            for (uint32_t c = 0; c < (1 << 8); c++)
            {
                if (std::find(bs.begin(), bs.end(), c) == bs.end())
                {
                    byte_map[get_unicode((1 << 8) + n)] = c;
                    ++n;
                }
            }

            return byte_map;
        }

        std::unordered_map<uint8_t, std::string> BPETokenizer::get_inv_byte_map()
        {
            auto byte_map = get_byte_map();
            std::unordered_map<uint8_t, std::string> inv_byte_map;
            for (auto &[code, byte] : byte_map)
            {
                inv_byte_map[byte] = code;
            }
            return inv_byte_map;
        }

        BPETokenizer BPETokenizer::from_file(const std::string &tokenizer_path)
        {
            return BPETokenizer(tokenizer_path);
        }

        std::vector<uint32_t> BPETokenizer::encode(const std::string &text, bool special_token) const
        {
            if (text.empty())
            {
                throw std::runtime_error("Cannot encode an empty string.");
            }
            std::vector<int> vis(text.size(), -1);
            size_t cnt = 0;
            for (size_t i = (special_token ? 0 : num_special_tokens); i < sorted_vocab.size(); i++)
            {
                auto &[token, id] = sorted_vocab[i];
                for (int i = 0; i + token.size() <= text.size(); i++)
                {
                    bool ok = true;
                    for (size_t j = 0; j < token.size(); j++)
                    {
                        if (vis[i + j] != -1 || text[i + j] != token[j])
                        {
                            ok = false;
                            break;
                        }
                    }
                    if (ok)
                    {
                        cnt += token.size();
                        vis[i] = id;
                        for (size_t j = 1; j < token.size(); j++)
                        {
                            vis[i + j] = -2; // Mark as part of a token
                        }
                        i += token.size() - 1; // Skip ahead
                    }
                }
                if (cnt == text.size())
                {
                    break; // All characters matched
                }
            }

            if (cnt < text.size())
            {
                throw std::runtime_error("Text could not be fully tokenized.");
            }
            std::vector<uint32_t> ids;
            for (size_t i = 0; i < vis.size(); i++)
            {
                if (vis[i] >= 0)
                {
                    ids.push_back(vis[i]);
                }
            }
            return ids;
        }

        std::string BPETokenizer::decode(const std::vector<uint32_t> &ids) const
        {
            std::string text;
            for (auto id : ids)
            {
                if (id < vocab.size())
                {
                    text += vocab[id];
                }
                else
                {
                    throw std::out_of_range("Token ID out of range: " + std::to_string(id));
                }
            }
            return text;
        }

        std::string BPETokenizer::decode(uint32_t id) const
        {
            if (id >= vocab.size())
            {
                throw std::out_of_range("Token ID out of range: " + std::to_string(id));
            }
            return vocab[id];
        }

        std::string BPETokenizer::decode(base::Tensor ids) const
        {
            base::Device device = ids.device();
            ids.toDevice(base::Device::CPU);

            std::vector<uint32_t> id_vec;
            size_t size = ids.size();
            uint32_t *id_data = reinterpret_cast<uint32_t *>(ids.data());
            for (size_t i = 0; i < size; i++)
            {
                id_vec.push_back(id_data[i]);
            }

            ids.toDevice(device);
            return decode(id_vec);
        }

        std::vector<uint32_t> BPETokenizer::encode_with_chat_template(
            const std::string &text,
            bool add_generation_prompt,
            bool enable_thinking) const
        {
            std::vector<uint32_t> token_ids({BPETokenizer::QWEN3_IM_START,
                                             BPETokenizer::QWEN3_USER,
                                             BPETokenizer::QWEN3_ENTER});
            auto text_ids = this->encode(text);
            token_ids.insert(token_ids.end(), text_ids.begin(), text_ids.end());
            token_ids.push_back(BPETokenizer::QWEN3_IM_END);
            token_ids.push_back(BPETokenizer::QWEN3_ENTER);
            if (add_generation_prompt)
            {
                token_ids.push_back(BPETokenizer::QWEN3_IM_START);
                token_ids.push_back(BPETokenizer::QWEN3_ASSISTANT);
                token_ids.push_back(BPETokenizer::QWEN3_ENTER);
                if (!enable_thinking)
                {
                    token_ids.push_back(BPETokenizer::QWEN3_THINK);
                    token_ids.push_back(271);
                    token_ids.push_back(BPETokenizer::QWEN3_END_THINK);
                    token_ids.push_back(271);
                }
            }
            return token_ids;
        }

        base::Tensor BPETokenizer::to_tensor(std::vector<uint32_t> ids, base::Device device) const
        {
            return base::Tensor::from_vector(ids, {ids.size(), 1}, device, false, nullptr);
        }
    } // namespace tokenizer
} // namespace mllm
