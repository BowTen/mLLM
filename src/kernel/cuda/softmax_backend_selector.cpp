#include "kernel/cuda/softmax_backend_selector.h"

#include <atomic>
#include <utility>

namespace mllm
{
    namespace kernel
    {
        namespace
        {
            std::atomic<int> last_softmax_backend_execution{
                static_cast<int>(SoftmaxBackendExecution::Unknown)};
            std::atomic<unsigned int> seen_softmax_backend_executions{0u};

            int to_storage(SoftmaxBackendExecution execution)
            {
                return static_cast<int>(execution);
            }

            unsigned int to_seen_bit(SoftmaxBackendExecution execution)
            {
                switch (execution)
                {
                case SoftmaxBackendExecution::HandwrittenFallback:
                    return 1u << 0;
                case SoftmaxBackendExecution::LibraryBacked:
                    return 1u << 1;
                case SoftmaxBackendExecution::Unknown:
                default:
                    return 0u;
                }
            }

            SoftmaxBackendExecution from_storage(int execution)
            {
                switch (execution)
                {
                case static_cast<int>(SoftmaxBackendExecution::HandwrittenFallback):
                    return SoftmaxBackendExecution::HandwrittenFallback;
                case static_cast<int>(SoftmaxBackendExecution::LibraryBacked):
                    return SoftmaxBackendExecution::LibraryBacked;
                case static_cast<int>(SoftmaxBackendExecution::Unknown):
                default:
                    return SoftmaxBackendExecution::Unknown;
                }
            }

            SoftmaxBackendSelection make_fallback(std::string reason)
            {
                return SoftmaxBackendSelection{SoftmaxBackend::HandwrittenFallback, std::move(reason)};
            }

            SoftmaxBackendSelection make_library_backed()
            {
                return SoftmaxBackendSelection{SoftmaxBackend::LibraryBacked, {}};
            }
        } // namespace

        SoftmaxBackendSelection select_softmax_backend(const base::Tensor &input,
                                                       const base::Tensor &output,
                                                       const SoftmaxBackendSelectionOptions &options)
        {
            if (!options.library_enabled)
            {
                return make_fallback("library-backed softmax is disabled");
            }

            if (!options.library_available)
            {
                return make_fallback("library-backed softmax is unavailable");
            }

            if (input.device() != base::Device::CUDA || output.device() != base::Device::CUDA)
            {
                return make_fallback("softmax backend selector requires CUDA tensors");
            }

            if (!input.is_contiguous() || !output.is_contiguous())
            {
                return make_fallback("softmax backend selector requires contiguous tensors");
            }

            if (input.shape() != output.shape())
            {
                return make_fallback("softmax input and output shapes must match");
            }

            if (input.shape().size() < 2 || input.shape().size() > 3)
            {
                return make_fallback("softmax library rollout supports rank 2 and rank 3 only");
            }

            return make_library_backed();
        }

        void record_last_softmax_backend_execution(SoftmaxBackendExecution execution)
        {
            last_softmax_backend_execution.store(to_storage(execution), std::memory_order_relaxed);
            seen_softmax_backend_executions.fetch_or(to_seen_bit(execution), std::memory_order_relaxed);
        }

        SoftmaxBackendExecution get_last_softmax_backend_execution()
        {
            return from_storage(last_softmax_backend_execution.load(std::memory_order_relaxed));
        }

        bool saw_softmax_backend_execution(SoftmaxBackendExecution execution)
        {
            const unsigned int execution_bit = to_seen_bit(execution);
            if (execution_bit == 0u)
            {
                return false;
            }

            return (seen_softmax_backend_executions.load(std::memory_order_relaxed) & execution_bit) != 0u;
        }

        void reset_last_softmax_backend_execution()
        {
            last_softmax_backend_execution.store(to_storage(SoftmaxBackendExecution::Unknown),
                                                 std::memory_order_relaxed);
            seen_softmax_backend_executions.store(0u, std::memory_order_relaxed);
        }
    } // namespace kernel
} // namespace mllm
