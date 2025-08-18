#include "base/safetensors.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        SafeTensors::SafeTensors(std::string file_path)
            : fd(-1), addr(nullptr), weight(nullptr)
        {
            // 打开文件
            fd = open(file_path.c_str(), O_RDONLY);
            if (fd == -1)
            {
                throw std::runtime_error("Failed to open file: " + file_path);
            }

            // 获取文件大小
            struct stat st;
            if (fstat(fd, &st) == -1)
            {
                close(fd);
                throw std::runtime_error("Failed to get file size");
            }

            // mmap整个文件
            addr = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr == MAP_FAILED)
            {
                close(fd);
                throw std::runtime_error("Failed to mmap file");
            }

            // 读取header长度（前8字节）
            if (st.st_size < 8)
            {
                munmap(addr, st.st_size);
                close(fd);
                throw std::runtime_error("File too small to contain header length");
            }

            uint64_t header_size = *reinterpret_cast<const uint64_t *>(addr);

            // 读取header（json格式）
            if (static_cast<uint64_t>(st.st_size) < 8 + header_size)
            {
                munmap(addr, st.st_size);
                close(fd);
                throw std::runtime_error("File too small to contain complete header");
            }

            const char *header_data = reinterpret_cast<const char *>(addr) + 8;
            std::string header_str(header_data, header_size);

            try
            {
                header = json::parse(header_str);
            }
            catch (const json::exception &e)
            {
                munmap(addr, st.st_size);
                close(fd);
                throw std::runtime_error("Failed to parse header JSON: " + std::string(e.what()));
            }

            // 权重数据起始位置
            weight = reinterpret_cast<char *>(addr) + 8 + header_size;
        }

        SafeTensors::~SafeTensors()
        {
            if (addr != nullptr && addr != MAP_FAILED)
            {
                // 获取文件大小来正确释放mmap
                struct stat st;
                if (fstat(fd, &st) == 0)
                {
                    munmap(addr, st.st_size);
                }
            }
            if (fd != -1)
            {
                close(fd);
            }
        }

        json SafeTensors::get_header() const
        {
            return header;
        }

        void *SafeTensors::get_weight(std::string weight_name) const
        {
            CHECK(weight != nullptr);

            CHECK(header.find(weight_name) != header.end()) << "Weight name not found in header: " << weight_name;

            const auto &weight_info = header[weight_name];
            CHECK(weight_info.find("data_offsets") != weight_info.end());

            const auto &offsets = weight_info["data_offsets"];
            CHECK(offsets.size() == 2);

            uint64_t start_offset = offsets[0];
            return reinterpret_cast<char *>(weight) + start_offset;
        }
    }
}