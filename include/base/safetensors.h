#ifndef MLLM_BASE_SAFETENSORS_H
#define MLLM_BASE_SAFETENSORS_H

#include "common.h"
#include "json.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mllm
{
    namespace base
    {
        using json = nlohmann::json;

        class SafeTensors
        {
            struct FileView;

            json header_;
            std::string root_dir_;
            std::unique_ptr<FileView> primary_file_;
            mutable std::unordered_map<std::string, std::unique_ptr<FileView>> shard_files_;
            std::unordered_map<std::string, std::string> weight_to_shard_;

            const FileView &resolve_file(const std::string &weight_name) const;
            const json &resolve_weight_info(const std::string &weight_name) const;
            const FileView &open_shard_file(const std::string &relative_path) const;

        public:
            SafeTensors(std::string file_path);
            ~SafeTensors();

            json get_header() const;
            std::vector<size_t> get_weight_shape(std::string weight_name) const;
            DType get_weight_dtype(std::string weight_name) const;
            void *get_weight(std::string weight_name) const;
            void materialize_weight(std::string weight_name, void *dst, DType target_dtype) const;
        };

    }
}

#endif
