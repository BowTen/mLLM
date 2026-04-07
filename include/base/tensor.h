#ifndef MLLM_BASE_TENSOR_H
#define MLLM_BASE_TENSOR_H

#include <iostream>
#include <vector>
#include <armadillo>
#include "allocator.h"
#include "buffer.h"
#include "common.h"
#include <cuda_runtime.h>
#include <type_traits>

#define GLOG_USE_GLOG_EXPORT
#include <glog/logging.h>

namespace mllm
{
    namespace base
    {
        class Tensor;

        class TensorMeta
        {
        private:
            std::vector<size_t> shape_;
            std::vector<size_t> stride_;
            bool is_contiguous_;
            Buffer::BufferPtr buffer_;
            Device device_;
            DType dtype_;
            bool mut_;
            size_t num_mats_ = 0;
            size_t size_;

            cudaStream_t stream_;

            static std::vector<size_t> default_stride(const std::vector<size_t> &shape);
            void update();

            friend Tensor;

        public:
            TensorMeta();
            TensorMeta(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device, DType dtype, bool mut, cudaStream_t stream);
            TensorMeta(const std::vector<size_t> &shape, Device device, bool mut, cudaStream_t stream, DType dtype = default_float_dtype());
            TensorMeta(void *data, const std::vector<size_t> &shape, bool copy, Device device, bool mut, cudaStream_t stream, DType dtype = default_float_dtype());
        };

        class Tensor
        {
        private:
            std::shared_ptr<TensorMeta> meta_;

            size_t check_index(int idx) const;

            template <class T>
            static DType infer_dtype_from_cpp_type()
            {
                if constexpr (std::is_same_v<T, float>)
                    return DType::FP32;
                else if constexpr (std::is_same_v<T, uint16_t>)
                    return DType::BF16;
                else if constexpr (std::is_same_v<T, uint32_t>)
                    return DType::U32;
                else
                    static_assert(!std::is_same_v<T, T>, "Unsupported Tensor::from_vector element type.");
            }

        public:
            Tensor() : meta_(std::make_shared<TensorMeta>()) {}
            Tensor(std::shared_ptr<TensorMeta> meta) : meta_(meta) {}
            Tensor(const std::vector<size_t> &shape, Buffer::BufferPtr buffer, Device device, bool mut, cudaStream_t stream, DType dtype = default_float_dtype())
                : meta_(std::make_shared<TensorMeta>(shape, buffer, device, dtype, mut, stream)) {}
            Tensor(const std::vector<size_t> &shape, Device device, bool mut, cudaStream_t stream, DType dtype = default_float_dtype())
                : meta_(std::make_shared<TensorMeta>(shape, device, mut, stream, dtype)) {}
            Tensor(void *data, const std::vector<size_t> &shape, bool copy, Device device, bool mut, cudaStream_t stream, DType dtype = default_float_dtype())
                : meta_(std::make_shared<TensorMeta>(data, shape, copy, device, mut, stream, dtype)) {}
            static Tensor rand(const std::vector<size_t> &shape, Device device, bool mut, cudaStream_t stream);
            static Tensor from_float(float value, Device device, bool mut, cudaStream_t stream);

            template <class T>
            static Tensor from_vector(std::vector<T> vec, std::vector<size_t> shape, Device device, bool mut, cudaStream_t stream)
            {
                const DType storage_dtype = infer_dtype_from_cpp_type<T>();
                Tensor tensor(vec.data(), shape, true, Device::CPU, mut, stream, storage_dtype);
                tensor.toDevice(device);
                return tensor;
            }

            void view(std::vector<size_t> shape);
            void reshape(std::vector<size_t> shape);
            void contiguous();
            void transpose(int i, int j);
            void t();

            float *operator[](size_t idx);
            float *operator[](std::vector<size_t> idx);
            size_t num_mats()
            {
                meta_->update();
                return meta_->num_mats_;
            }
            float *mat(size_t idx);

            const std::vector<size_t> &shape() const { return meta_->shape_; }
            size_t shape(int idx) const;
            size_t stride(int idx) const;
            const std::vector<size_t> &stride() const { return meta_->stride_; }
            void set_buffer(Buffer::BufferPtr buffer) { meta_->buffer_ = buffer; }
            Buffer::BufferPtr buffer() const { return meta_->buffer_; }
            bool is_contiguous() const { return meta_->is_contiguous_; }
            size_t logic_size() const { return meta_->size_; }
            size_t size() const
            {
                if (!meta_->buffer_)
                    return 0;
                return meta_->buffer_->size() / element_size();
            }
            void *raw_data();
            const void *raw_data() const;

            template <class T>
            T *data()
            {
                if (infer_dtype_from_cpp_type<T>() != meta_->dtype_)
                    return nullptr;
                return static_cast<T *>(raw_data());
            }

            template <class T>
            const T *data() const
            {
                if (infer_dtype_from_cpp_type<T>() != meta_->dtype_)
                    return nullptr;
                return static_cast<const T *>(raw_data());
            }

            float *compatible_float_data();
            const float *compatible_float_data() const;
            float *data() { return compatible_float_data(); } // Compatibility path for legacy FP32 callers.
            const float *data() const { return compatible_float_data(); }
            bool empty() const { return meta_->buffer_ == nullptr; }
            Tensor toDevice(Device device);
            Tensor clone();

            void push(const void *bytes, size_t num_bytes);
            void cat(Tensor &other, int dim_int);

            void insert_dim(int dim_int);
            void expand(int dim_int, size_t size);
            void reserve(size_t size);
            void resize(size_t size);

            std::vector<size_t> index(size_t id);

            base::Device device() const { return meta_->device_; }
            DType dtype() const { return meta_->dtype_; }
            size_t element_size() const { return dtype_element_size(meta_->dtype_); }
            cudaStream_t stream() const { return meta_->stream_; }
            void set_stream(cudaStream_t stream) { meta_->stream_ = stream; }
        };

        using PosEmb = std::pair<Tensor *, Tensor *>; // cos sin
    } // namespace base
} // namespace mllm

#endif // MLLM_BASE_TENSOR_H
