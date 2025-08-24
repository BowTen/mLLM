#ifndef MLLM_BASE_SAFETENSORS_H
#define MLLM_BASE_SAFETENSORS_H

#include "json.hpp"
#include <string>

namespace mllm
{
    namespace base
    {
        using json = nlohmann::json;

        class SafeTensors
        {
            int fd;
            void *addr;
            json header;
            void *weight;

        public:
            SafeTensors(std::string file_path);
            ~SafeTensors();

            json get_header() const;
            std::vector<size_t> get_weight_shape(std::string weight_name) const;
            void *get_weight(std::string weight_name) const;
        };

    }
}

#endif