#ifndef MLLM_BASE_COMMON_H
#define MLLM_BASE_COMMON_H

#include <utility>

namespace mllm
{
    namespace base
    {
#define DEBUG 1
#define TRACE 2

        enum Device
        {
            CPU = 0,
            CUDA = 1,
        };

        bool isDevicePointer(void *ptr);

#define FLOAT_INF 1e9
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_COMMON_H