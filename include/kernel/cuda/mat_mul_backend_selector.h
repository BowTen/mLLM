#ifndef MLLM_KERNEL_CUDA_MAT_MUL_BACKEND_SELECTOR_H
#define MLLM_KERNEL_CUDA_MAT_MUL_BACKEND_SELECTOR_H

#include <string>
#include "base/tensor.h"

namespace mllm
{
    namespace kernel
    {
        enum class MatMulBackend
        {
            HandwrittenFallback,
            LibraryBacked,
        };

        enum class MatMulBackendExecution
        {
            Unknown,
            HandwrittenFallback,
            LibraryBacked,
        };

        struct MatMulBackendSelection
        {
            MatMulBackend backend = MatMulBackend::HandwrittenFallback;
            std::string reason;

            bool uses_library_backend() const
            {
                return backend == MatMulBackend::LibraryBacked;
            }
        };

        struct MatMulBackendSelectionOptions
        {
            bool library_enabled = false;
            bool library_available = false;
            bool allow_batched_matmul = false;
        };

        // Process-wide last-writer-wins execution trace for matmul observability.
        // Reset before a benchmark/debug window and query after the invocation(s)
        // you want to inspect. Reset also clears the process-wide "seen in this
        // window" flags used by validation and benchmark workflows. The last
        // execution value reflects the most recent host-side execution path
        // recorded by any thread.
        MatMulBackendSelection select_mat_mul_backend(const base::Tensor &input0,
                                                       const base::Tensor &input1,
                                                       const base::Tensor &output,
                                                       const MatMulBackendSelectionOptions &options = {});

        void record_last_mat_mul_backend_execution(MatMulBackendExecution execution);
        MatMulBackendExecution get_last_mat_mul_backend_execution();
        bool saw_mat_mul_backend_execution(MatMulBackendExecution execution);
        void reset_last_mat_mul_backend_execution();
    } // namespace kernel
} // namespace mllm

#endif // MLLM_KERNEL_CUDA_MAT_MUL_BACKEND_SELECTOR_H
