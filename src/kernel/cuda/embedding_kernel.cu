#include "kernel/cuda/embedding_kernel.cuh"
#include "kernel/kernel.h"
#include "base/util.h"

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace kernel
    {
        __global__ void emb_kernel_cuda_fp32(uint32_t *input_data,
                                             float *weight_data,
                                             float *output_data,
                                             uint32_t seq_len,
                                             uint32_t hidden_size)
        {
            uint32_t mat_id = blockIdx.x;
            uint32_t token_id = blockIdx.y;
            uint32_t emb_id = input_data[mat_id * seq_len + token_id];
            weight_data += emb_id * hidden_size;
            output_data += mat_id * seq_len * hidden_size + token_id * hidden_size;

            uint32_t vec_size = hidden_size / 4;
            uint32_t vec_end = vec_size * 4;

            float4 *weight_data_vec = reinterpret_cast<float4 *>(weight_data);
            float4 *output_data_vec = reinterpret_cast<float4 *>(output_data);

            for (uint32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                output_data_vec[i] = weight_data_vec[i];
            }
            for (uint32_t i = vec_end + threadIdx.x; i < hidden_size; i += blockDim.x)
            {
                output_data[i] = weight_data[i];
            }
        }

        __device__ void copy(float *dst, float *src, uint32_t len)
        {
            int32_t vec_size = len / 4;
            int32_t vec_end = vec_size * 4;
            float4 *dst_vec = reinterpret_cast<float4 *>(dst);
            float4 *src_vec = reinterpret_cast<float4 *>(src);
            for (int32_t i = threadIdx.x; i < vec_size; i += blockDim.x)
            {
                dst_vec[i] = src_vec[i];
            }
            for (int32_t i = vec_end + threadIdx.x; i < len; i += blockDim.x)
            {
                dst[i] = src[i];
            }
        }

        __global__ void router(uint32_t *input, float *weight, float *output, uint32_t seq_len, uint32_t hidden_size)
        {
            uint32_t mat_id = blockIdx.x;
            uint32_t token_id = blockIdx.y;
            uint32_t emb_id = input[mat_id * seq_len + token_id];

            copy(output + mat_id * seq_len * hidden_size + token_id * hidden_size, weight + emb_id * hidden_size, hidden_size);
        }

        void emb_kernel_cuda(base::Tensor *input,
                             base::Tensor *weight,
                             base::Tensor *output,
                             size_t hidden_size,
                             void *stream)
        {
            CHECK(input->shape(-2) == output->shape(-2));
            input->contiguous();
            weight->contiguous();
            output->contiguous();

            uint32_t *input_data = (uint32_t *)input->data();
            float *weight_data = (float *)weight->data();
            float *output_data = (float *)output->data();
            size_t seq_len = input->shape(-2);

            dim3 grid(input->num_mats(), seq_len);
            VLOG(DEBUG) << "Launching CUDA embedding kernel with grid size: " << grid.x << ", " << grid.y;

            if (stream)
            {
                emb_kernel_cuda_fp32<<<grid, 128, 0, (cudaStream_t)stream>>>(
                    input_data,
                    weight_data,
                    output_data,
                    seq_len,
                    hidden_size);
            }
            else
            {
                LOG(WARNING) << "CUDA stream is null, using default stream.";
                emb_kernel_cuda_fp32<<<grid, 128>>>(
                    input_data,
                    weight_data,
                    output_data,
                    seq_len,
                    hidden_size);
            }
            CHECK_CUDA_ERR(cudaDeviceSynchronize());

            CHECK_CUDA_ERR(cudaGetLastError());
        }
    }
}