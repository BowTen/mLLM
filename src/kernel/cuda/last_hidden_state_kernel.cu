#include "kernel/cpu/last_hidden_state_kernel.h"
#include "base/allocator.h"
#include "base/util.h"

namespace mllm
{
    namespace kernel
    {
        void last_hidden_state_kernel_cuda(base::Tensor *input,
                                           void *stream)
        {
            if (input->shape(-2) == 1)
            {
                return;
            }
            input->contiguous();

            auto shape = input->shape();
            size_t hidden_size = shape.back();
            size_t num_mats = input->num_mats();
            size_t seq_len = input->shape(-2);
            float *new_data = input->data();
            base::Allocator *allocator = base::CudaAllocator::getInstance();
            for (size_t i = 0; i < num_mats; i++)
            {
                float *input_row = input->mat(i) + (seq_len - 1) * hidden_size;
                allocator->memcpy(new_data, input_row, hidden_size * sizeof(float));
                new_data += hidden_size;
            }
            shape[shape.size() - 2] = 1;
            input->view(shape);
            // CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}