#include "base/util.h"
#include "base/allocator.h"
#include <fstream>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        json load_json(const std::string &file_path)
        {
            std::ifstream file(file_path);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file: " + file_path);
            }
            json j;
            file >> j;
            return j;
        }

        void load_bf16_to_f32(const void *src, void *dst, size_t num_elements)
        {
            CHECK(src);
            CHECK(dst);
            auto allocator = HostAllocator::getInstance();
            allocator->memcpy(dst, src, num_elements * sizeof(uint16_t));
            uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
            uint32_t *dst_fp32 = static_cast<uint32_t *>(dst);
            for (int i = num_elements - 1; i >= 0; i--)
            {
                dst_fp32[i] = (dst_bf16[i] << 16);
            }
        }

        std::mt19937 global_mt(std::random_device{}());
        std::uniform_real_distribution<> global_urd(0.0, 1.0);

        float get_random_float()
        {
            return global_urd(global_mt);
        }
    }
}