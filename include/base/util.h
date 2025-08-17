#ifndef MLLM_BASE_UTIL_H
#define MLLM_BASE_UTIL_H

#include "json.hpp"

namespace mllm
{
    namespace base
    {
        using json = nlohmann::json;

        json load_json(const std::string &file_path);

        void load_bf16_to_f32(const void *src, void *dst, size_t num_elements);
    }
}

#endif // MLLM_BASE_UTIL_H