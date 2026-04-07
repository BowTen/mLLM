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

        void load_f32_to_bf16(const void *src, void *dst, size_t num_elements)
        {
            CHECK(src);
            CHECK(dst);
            const uint32_t *src_fp32 = static_cast<const uint32_t *>(src);
            uint16_t *dst_bf16 = static_cast<uint16_t *>(dst);
            for (size_t i = 0; i < num_elements; ++i)
            {
                dst_bf16[i] = static_cast<uint16_t>(src_fp32[i] >> 16);
            }
        }

        void materialize_float_storage(const void *src, DType src_dtype, void *dst, DType dst_dtype, size_t num_elements)
        {
            CHECK(src);
            CHECK(dst);
            CHECK(is_floating_point_dtype(src_dtype)) << "Source dtype must be floating point.";
            CHECK(is_floating_point_dtype(dst_dtype)) << "Destination dtype must be floating point.";

            if (src_dtype == dst_dtype)
            {
                HostAllocator::getInstance()->memcpy(dst, src, num_elements * dtype_element_size(src_dtype));
                return;
            }

            if (src_dtype == DType::BF16 && dst_dtype == DType::FP32)
            {
                load_bf16_to_f32(src, dst, num_elements);
                return;
            }

            if (src_dtype == DType::FP32 && dst_dtype == DType::BF16)
            {
                load_f32_to_bf16(src, dst, num_elements);
                return;
            }

            CHECK(false) << "Unsupported float storage conversion.";
        }

        std::mt19937 global_mt(time(0));
        std::uniform_real_distribution<> global_urd(0.0, 1.0);

        float get_random_float()
        {
            return global_urd(global_mt);
        }
    }
}
