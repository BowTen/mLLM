#include <iostream>
#include <vector>
#include "allocator.h"
#include "buffer.h"

namespace mllm
{
    namespace base
    {
        enum Device
        {
            CPU = 0,
            CUDA = 1,
        };

        class Tensor
        {
            std::vector<size_t> shape_;
            Buffer::BufferPtr buffer_;
            Device device_;

        public:
            Tensor(const std::vector<size_t> &shape, Device device, bool mut = false);

            const std::vector<size_t> &shape() const;
            size_t size() const;
            float *data();
        };
    } // namespace base
} // namespace mllm