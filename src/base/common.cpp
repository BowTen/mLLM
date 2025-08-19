#include "base/common.h"
#include <cuda_runtime.h>

namespace mllm
{
    namespace base
    {

        bool isDevicePointer(void *ptr)
        {
            cudaPointerAttributes attributes;
            cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

            if (err == cudaSuccess)
            {
                return (attributes.type == cudaMemoryTypeDevice);
            }
            return false;
        }
    }
}