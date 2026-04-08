#include "base/safetensors.h"
#include "base/util.h"
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        namespace
        {
            bool is_index_manifest_path(const std::string &file_path)
            {
                constexpr const char kSuffix[] = ".index.json";
                const size_t suffix_size = sizeof(kSuffix) - 1;
                return file_path.size() >= suffix_size &&
                       file_path.compare(file_path.size() - suffix_size, suffix_size, kSuffix) == 0;
            }
        } // namespace

        struct SafeTensors::FileView
        {
            int fd = -1;
            void *addr = nullptr;
            size_t file_size = 0;
            json header;
            void *weight = nullptr;

            explicit FileView(const std::string &file_path)
            {
                fd = open(file_path.c_str(), O_RDONLY);
                if (fd == -1)
                {
                    throw std::runtime_error("Failed to open file: " + file_path);
                }

                struct stat st;
                if (fstat(fd, &st) == -1)
                {
                    close(fd);
                    throw std::runtime_error("Failed to get file size");
                }
                file_size = static_cast<size_t>(st.st_size);

                addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
                if (addr == MAP_FAILED)
                {
                    close(fd);
                    throw std::runtime_error("Failed to mmap file");
                }

                if (file_size < sizeof(uint64_t))
                {
                    munmap(addr, file_size);
                    close(fd);
                    throw std::runtime_error("File too small to contain header length");
                }

                const uint64_t header_size = *reinterpret_cast<const uint64_t *>(addr);
                if (file_size < sizeof(uint64_t) + header_size)
                {
                    munmap(addr, file_size);
                    close(fd);
                    throw std::runtime_error("File too small to contain complete header");
                }

                const char *header_data = reinterpret_cast<const char *>(addr) + sizeof(uint64_t);
                const std::string header_str(header_data, header_size);

                try
                {
                    header = json::parse(header_str);
                }
                catch (const json::exception &e)
                {
                    munmap(addr, file_size);
                    close(fd);
                    throw std::runtime_error("Failed to parse header JSON: " + std::string(e.what()));
                }

                weight = reinterpret_cast<char *>(addr) + sizeof(uint64_t) + header_size;
            }

            ~FileView()
            {
                if (addr != nullptr && addr != MAP_FAILED)
                {
                    munmap(addr, file_size);
                }
                if (fd != -1)
                {
                    close(fd);
                }
            }
        };

        SafeTensors::SafeTensors(std::string file_path)
            : header_(json::object()),
              root_dir_(std::filesystem::path(file_path).parent_path().string())
        {
            if (is_index_manifest_path(file_path))
            {
                std::ifstream manifest_stream(file_path);
                if (!manifest_stream.is_open())
                {
                    throw std::runtime_error("Failed to open file: " + file_path);
                }
                try
                {
                    header_ = json::parse(manifest_stream);
                }
                catch (const json::exception &e)
                {
                    throw std::runtime_error("Failed to parse manifest JSON: " + std::string(e.what()));
                }

                CHECK(header_.contains("weight_map")) << "Missing weight_map in manifest: " << file_path;
                for (const auto &[weight_name, shard_name] : header_["weight_map"].items())
                {
                    weight_to_shard_[weight_name] = shard_name.get<std::string>();
                }
                return;
            }

            primary_file_ = std::make_unique<FileView>(file_path);
            header_ = primary_file_->header;
        }

        SafeTensors::~SafeTensors()
        {
        }

        json SafeTensors::get_header() const
        {
            return header_;
        }

        const SafeTensors::FileView &SafeTensors::open_shard_file(const std::string &relative_path) const
        {
            const std::filesystem::path shard_path = std::filesystem::path(root_dir_) / relative_path;
            const std::string shard_key = shard_path.string();
            auto it = shard_files_.find(shard_key);
            if (it == shard_files_.end())
            {
                auto inserted = shard_files_.emplace(shard_key, std::make_unique<FileView>(shard_key));
                it = inserted.first;
            }
            return *it->second;
        }

        const SafeTensors::FileView &SafeTensors::resolve_file(const std::string &weight_name) const
        {
            if (weight_to_shard_.empty())
            {
                CHECK(primary_file_ != nullptr);
                return *primary_file_;
            }

            const auto shard_it = weight_to_shard_.find(weight_name);
            CHECK(shard_it != weight_to_shard_.end()) << "Weight name not found in manifest: " << weight_name;
            return open_shard_file(shard_it->second);
        }

        const json &SafeTensors::resolve_weight_info(const std::string &weight_name) const
        {
            const auto &file = resolve_file(weight_name);
            CHECK(file.header.find(weight_name) != file.header.end()) << "Weight name not found in header: " << weight_name;
            return file.header.at(weight_name);
        }

        DType SafeTensors::get_weight_dtype(std::string weight_name) const
        {
            const auto &weight_info = resolve_weight_info(weight_name);
            CHECK(weight_info.find("dtype") != weight_info.end()) << "Weight dtype not found in header: " << weight_name;
            const std::string dtype = weight_info["dtype"];
            if (dtype == "F32")
                return DType::FP32;
            if (dtype == "BF16")
                return DType::BF16;
            CHECK(false) << "Unsupported safetensors dtype: " << dtype;
        }

        void *SafeTensors::get_weight(std::string weight_name) const
        {
            const auto &file = resolve_file(weight_name);
            CHECK(file.weight != nullptr);
            const auto &weight_info = resolve_weight_info(weight_name);
            CHECK(weight_info.find("data_offsets") != weight_info.end());

            const auto &offsets = weight_info["data_offsets"];
            CHECK(offsets.size() == 2);

            uint64_t start_offset = offsets[0];
            return reinterpret_cast<char *>(file.weight) + start_offset;
        }

        void SafeTensors::materialize_weight(std::string weight_name, void *dst, DType target_dtype) const
        {
            CHECK(dst != nullptr);
            const DType source_dtype = get_weight_dtype(weight_name);
            const auto weight_shape = get_weight_shape(weight_name);
            size_t num_elements = 1;
            for (size_t dim : weight_shape)
                num_elements *= dim;
            materialize_float_storage(get_weight(weight_name), source_dtype, dst, target_dtype, num_elements);
        }

        std::vector<size_t> SafeTensors::get_weight_shape(std::string weight_name) const
        {
            const auto &weight_info = resolve_weight_info(weight_name);
            CHECK(weight_info.find("shape") != weight_info.end());

            const auto &shape = weight_info["shape"];

            return shape;
        }

    }
}
