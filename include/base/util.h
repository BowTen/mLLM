#ifndef MLLM_BASE_UTIL_H
#define MLLM_BASE_UTIL_H

#include "json.hpp"
#include <random>
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

#define CHECK_CUDA_ERR(err)                                            \
    do                                                                 \
    {                                                                  \
        cudaError_t status = (err);                                    \
        if (status != cudaSuccess)                                     \
        {                                                              \
            CHECK(false) << "CUDA error: " << cudaGetErrorString(status); \
        }                                                              \
    } while (0)

namespace mllm
{
    namespace base
    {
        using json = nlohmann::json;

        json load_json(const std::string &file_path);

        void load_bf16_to_f32(const void *src, void *dst, size_t num_elements);

        float get_random_float();

    }
}

#endif // MLLM_BASE_UTIL_H
