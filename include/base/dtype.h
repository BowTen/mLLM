#ifndef MLLM_BASE_DTYPE_H
#define MLLM_BASE_DTYPE_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace mllm
{
    namespace base
    {
        enum class DType
        {
            FP32,
            BF16,
            U32,
        };

        inline DType default_float_dtype()
        {
            return DType::FP32;
        }

        inline bool is_floating_point_dtype(DType dtype)
        {
            return dtype == DType::FP32 || dtype == DType::BF16;
        }

        inline size_t dtype_element_size(DType dtype)
        {
            switch (dtype)
            {
            case DType::FP32:
                return sizeof(float);
            case DType::BF16:
                return sizeof(uint16_t);
            case DType::U32:
                return sizeof(uint32_t);
            default:
                throw std::invalid_argument("Unsupported dtype.");
            }
        }
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_DTYPE_H
