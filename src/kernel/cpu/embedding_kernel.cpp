#include "kernel/cpu/embedding_kernel.h"
#include <cuda_runtime.h>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        void mat_emb_kernel_cpu(uint32_t *input_data,
                                float *weight_data,
                                float *output_data,
                                size_t sqe_size,
                                size_t hidden_size)
        {
            auto allocator = base::HostAllocator::getInstance();
            for (size_t i = 0; i < sqe_size; ++i)
            {
                size_t emb_id = input_data[i];
                VLOG(TRACE) << "Processing embedding for input ID: " << emb_id;
                allocator->memcpy(output_data + i * hidden_size,
                                  weight_data + emb_id * hidden_size,
                                  hidden_size * sizeof(float));
            }
        }
        void emb_kernel_cpu(base::Tensor *input,
                            base::Tensor *weight,
                            base::Tensor *output,
                            size_t hidden_size,
                            [[maybe_unused]] void *stream)
        {
            VLOG(DEBUG) << "input[0]: " << reinterpret_cast<uint32_t *>(input->data())[0];
            CHECK(input->shape(-2) == output->shape(-2));
            float *weight_data = weight->data();

            size_t sqe_size = input->shape(-2);
            size_t num_mats = input->num_mats();
            for (size_t i = 0; i < num_mats; i++)
            {
                VLOG(DEBUG) << "Embedding kernel, processing matrix " << i;
                mat_emb_kernel_cpu(reinterpret_cast<uint32_t *>(input->mat(i)),
                                   weight_data,
                                   reinterpret_cast<float *>(output->mat(i)),
                                   sqe_size,
                                   hidden_size);
            }
        }
    }
}