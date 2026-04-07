#include "kernel/cuda/mat_mul_backend_selector.h"

#include <atomic>
#include <sstream>
#include <string>

namespace mllm
{
    namespace kernel
    {
        namespace
        {
            std::atomic<int> last_mat_mul_backend_execution{
                static_cast<int>(MatMulBackendExecution::Unknown)};
            std::atomic<unsigned int> seen_mat_mul_backend_executions{0u};

            int to_storage(MatMulBackendExecution execution)
            {
                return static_cast<int>(execution);
            }

            unsigned int to_seen_bit(MatMulBackendExecution execution)
            {
                switch (execution)
                {
                case MatMulBackendExecution::HandwrittenFallback:
                    return 1u << 0;
                case MatMulBackendExecution::LibraryBacked:
                    return 1u << 1;
                case MatMulBackendExecution::Unknown:
                default:
                    return 0u;
                }
            }

            MatMulBackendExecution from_storage(int execution)
            {
                switch (execution)
                {
                case static_cast<int>(MatMulBackendExecution::HandwrittenFallback):
                    return MatMulBackendExecution::HandwrittenFallback;
                case static_cast<int>(MatMulBackendExecution::LibraryBacked):
                    return MatMulBackendExecution::LibraryBacked;
                case static_cast<int>(MatMulBackendExecution::Unknown):
                default:
                    return MatMulBackendExecution::Unknown;
                }
            }

            MatMulBackendSelection make_fallback(std::string reason)
            {
                return MatMulBackendSelection{MatMulBackend::HandwrittenFallback, std::move(reason)};
            }

            MatMulBackendSelection make_library_backed()
            {
                return MatMulBackendSelection{MatMulBackend::LibraryBacked, {}};
            }

            bool shapes_match_prefix(const std::vector<size_t> &lhs,
                                     const std::vector<size_t> &rhs,
                                     const std::vector<size_t> &out,
                                     size_t batch_dims,
                                     std::string &reason)
            {
                for (size_t i = 0; i < batch_dims; ++i)
                {
                    if (lhs[i] != rhs[i] || lhs[i] != out[i])
                    {
                        std::ostringstream oss;
                        oss << "batch dimension mismatch at axis " << i;
                        reason = oss.str();
                        return false;
                    }
                }
                return true;
            }
        } // namespace

        MatMulBackendSelection select_mat_mul_backend(const base::Tensor &input0,
                                                       const base::Tensor &input1,
                                                       const base::Tensor &output,
                                                       const MatMulBackendSelectionOptions &options)
        {
            if (!options.library_enabled)
            {
                return make_fallback("library-backed matmul is disabled");
            }

            if (!options.library_available)
            {
                return make_fallback("library-backed matmul is unavailable");
            }

            if (input1.size() == 1)
            {
                return make_fallback("scalar rhs uses handwritten fallback");
            }

            if (input0.device() != base::Device::CUDA || input1.device() != base::Device::CUDA || output.device() != base::Device::CUDA)
            {
                return make_fallback("matmul backend selector requires CUDA tensors");
            }

            if (!input0.is_contiguous() || !input1.is_contiguous() || !output.is_contiguous())
            {
                return make_fallback("matmul backend selector requires contiguous tensors");
            }

            const auto &lhs_shape = input0.shape();
            const auto &rhs_shape = input1.shape();
            const auto &out_shape = output.shape();

            if (lhs_shape.size() < 2)
            {
                return make_fallback("lhs rank must be at least 2");
            }
            if (rhs_shape.size() < 2)
            {
                return make_fallback("rhs rank must be at least 2");
            }
            if (out_shape.size() < 2)
            {
                return make_fallback("output rank must be at least 2");
            }

            if (lhs_shape.size() != rhs_shape.size() || lhs_shape.size() != out_shape.size())
            {
                return make_fallback("matmul tensors must have the same rank");
            }

            if (lhs_shape.size() > 2)
            {
                if (options.allow_batched_matmul)
                {
                    return make_fallback("library-backed matmul adapter is 2D-only");
                }
                return make_fallback("batched matmul is disabled for this rollout");
            }

            const size_t batch_dims = lhs_shape.size() - 2;
            std::string reason;
            if (!shapes_match_prefix(lhs_shape, rhs_shape, out_shape, batch_dims, reason))
            {
                return make_fallback(reason);
            }

            if (lhs_shape[lhs_shape.size() - 1] != rhs_shape[rhs_shape.size() - 2])
            {
                return make_fallback("lhs inner dimension must match rhs rows");
            }

            if (out_shape[out_shape.size() - 2] != lhs_shape[lhs_shape.size() - 2] || out_shape.back() != rhs_shape.back())
            {
                return make_fallback("output shape must match matmul result");
            }

            return make_library_backed();
        }

        void record_last_mat_mul_backend_execution(MatMulBackendExecution execution)
        {
            last_mat_mul_backend_execution.store(to_storage(execution), std::memory_order_relaxed);
            seen_mat_mul_backend_executions.fetch_or(to_seen_bit(execution), std::memory_order_relaxed);
        }

        MatMulBackendExecution get_last_mat_mul_backend_execution()
        {
            return from_storage(last_mat_mul_backend_execution.load(std::memory_order_relaxed));
        }

        bool saw_mat_mul_backend_execution(MatMulBackendExecution execution)
        {
            const unsigned int execution_bit = to_seen_bit(execution);
            if (execution_bit == 0u)
            {
                return false;
            }

            return (seen_mat_mul_backend_executions.load(std::memory_order_relaxed) & execution_bit) != 0u;
        }

        void reset_last_mat_mul_backend_execution()
        {
            last_mat_mul_backend_execution.store(to_storage(MatMulBackendExecution::Unknown),
                                                 std::memory_order_relaxed);
            seen_mat_mul_backend_executions.store(0u, std::memory_order_relaxed);
        }
    } // namespace kernel
} // namespace mllm
