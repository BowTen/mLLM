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
            CHECK(input->dtype() == base::DType::U32) << "Embedding input must use U32 token ids.";
            CHECK(input->shape(-2) == output->shape(-2));
            if (weight->dtype() != base::DType::FP32 || output->dtype() != base::DType::FP32)
            {
                base::Tensor weight_fp32 = weight->astype(base::DType::FP32);
                base::Tensor output_fp32(output->shape(), base::Device::CPU, output->is_mutable(), nullptr, base::DType::FP32);
                emb_kernel_cpu(input, &weight_fp32, &output_fp32, hidden_size, stream);
                *output = output_fp32.astype(output->dtype());
                return;
            }

            const uint32_t *input_data = input->data<uint32_t>();
            CHECK(input_data != nullptr) << "Embedding input token ids are unavailable.";
            float *weight_data = weight->data<float>();
            float *output_data = output->data<float>();
            VLOG(DEBUG) << "input[0]: " << input_data[0];

            size_t sqe_size = input->shape(-2);
            size_t num_mats = input->num_mats();
            for (size_t i = 0; i < num_mats; i++)
            {
                VLOG(DEBUG) << "Embedding kernel, processing matrix " << i;
                mat_emb_kernel_cpu(const_cast<uint32_t *>(input_data + i * sqe_size),
                                   weight_data,
                                   output_data + i * sqe_size * hidden_size,
                                   sqe_size,
                                   hidden_size);
            }
        }
    }
}
