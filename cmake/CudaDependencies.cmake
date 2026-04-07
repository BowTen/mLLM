include_guard(GLOBAL)

function(mllm_configure_cuda_dependencies)
    find_package(CUDAToolkit REQUIRED)

    find_library(MLLM_CUDNN_LIBRARY
        NAMES cudnn
        HINTS
            "${CUDAToolkit_LIBRARY_DIR}"
            "${CUDAToolkit_LIBRARY_ROOT}/lib64"
            "${CUDAToolkit_LIBRARY_ROOT}/lib"
            /usr/local/cuda/lib64
            /usr/lib/x86_64-linux-gnu
            /usr/lib64
    )
    find_path(MLLM_CUDNN_INCLUDE_DIR
        NAMES cudnn.h
        HINTS
            "${CUDAToolkit_INCLUDE_DIRS}"
            "${CUDAToolkit_LIBRARY_ROOT}/include"
            /usr/local/cuda/include
            /usr/include
    )

    if(MLLM_CUDNN_LIBRARY AND MLLM_CUDNN_INCLUDE_DIR)
        set(MLLM_CUDNN_AVAILABLE TRUE CACHE BOOL "Whether cuDNN is available" FORCE)
        set(MLLM_CUDNN_LIBRARY "${MLLM_CUDNN_LIBRARY}" CACHE FILEPATH "Path to the detected cuDNN library" FORCE)
        set(MLLM_CUDNN_INCLUDE_DIR "${MLLM_CUDNN_INCLUDE_DIR}" CACHE PATH "Path to the detected cuDNN headers" FORCE)
        message(STATUS "Optional CUDA library detected: cuDNN (${MLLM_CUDNN_LIBRARY})")
    else()
        set(MLLM_CUDNN_AVAILABLE FALSE CACHE BOOL "Whether cuDNN is available" FORCE)
        unset(MLLM_CUDNN_LIBRARY CACHE)
        unset(MLLM_CUDNN_INCLUDE_DIR CACHE)
        message(STATUS "Optional CUDA library not found: cuDNN; handwritten CUDA kernels remain available.")
    endif()
endfunction()

function(mllm_target_require_cuda_library target imported_target friendly_name)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Target '${target}' must exist before requiring CUDA library '${friendly_name}'.")
    endif()

    if(NOT TARGET "${imported_target}")
        message(FATAL_ERROR "Required CUDA library ${friendly_name} was not found; target '${target}' requires ${imported_target}.")
    endif()

    target_link_libraries("${target}" PRIVATE "${imported_target}")
    message(STATUS "Target '${target}' links required CUDA library: ${friendly_name} (${imported_target})")
endfunction()
