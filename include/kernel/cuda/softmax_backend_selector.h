#ifndef MLLM_KERNEL_CUDA_SOFTMAX_BACKEND_SELECTOR_H
#define MLLM_KERNEL_CUDA_SOFTMAX_BACKEND_SELECTOR_H

#include <string>

#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        enum class SoftmaxBackend
        {
            HandwrittenFallback,
            LibraryBacked,
        };

        enum class SoftmaxBackendExecution
        {
            Unknown,
            HandwrittenFallback,
            LibraryBacked,
        };

        struct SoftmaxBackendSelection
        {
            SoftmaxBackend backend = SoftmaxBackend::HandwrittenFallback;
            std::string reason;

            bool uses_library_backend() const
            {
                return backend == SoftmaxBackend::LibraryBacked;
            }
        };

        struct SoftmaxBackendSelectionOptions
        {
            bool library_enabled = false;
            bool library_available = false;
        };

        SoftmaxBackendSelection select_softmax_backend(const base::Tensor &input,
                                                       const base::Tensor &output,
                                                       const SoftmaxBackendSelectionOptions &options = {});

        void record_last_softmax_backend_execution(SoftmaxBackendExecution execution);
        SoftmaxBackendExecution get_last_softmax_backend_execution();
        bool saw_softmax_backend_execution(SoftmaxBackendExecution execution);
        void reset_last_softmax_backend_execution();
    } // namespace kernel
} // namespace mllm

#endif // MLLM_KERNEL_CUDA_SOFTMAX_BACKEND_SELECTOR_H
