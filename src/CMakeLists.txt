cmake_minimum_required(VERSION 3.14)

add_library(base 
    base/allocator.cpp
    base/buffer.cpp
    base/tensor.cpp
    base/safetensors.cpp
    base/util.cpp
)
add_library(tokenizer 
    tokenizer/tokenizer.cpp
)
add_library(op
    op/layer.cpp
    op/embedding.cpp
    op/rms_norm.cpp
    op/add.cpp
    op/linear.cpp
)
add_library(kernel
    kernel/kernel.cpp
    kernel/cpu/embedding_kernel.cpp
    kernel/cuda/embedding_kernel.cu
    kernel/cpu/rms_norm_kernel.cpp
    kernel/cuda/rms_norm_kernel.cu
    kernel/cpu/add_kernel.cpp
    kernel/cuda/add_kernel.cu
    kernel/cpu/mat_mul_kernel.cpp
    kernel/cuda/mat_mul_kernel.cu
)
add_library(model 
    model/qwen3.cpp
    model/qwen3_decode_layer.cpp
    model/qwen3_self_attn.cpp
    model/qwen3_mlp.cpp
)

# Link libraries
find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ARMADILLO REQUIRED armadillo)

target_link_libraries(base glog CUDA::cudart armadillo openblas)
target_link_libraries(tokenizer base)
target_link_libraries(kernel base armadillo openblas)
target_link_libraries(op base kernel)
target_link_libraries(model tokenizer op)


# Include directories for the libraries
target_include_directories(base PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(tokenizer PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(op PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(kernel PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(model PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# Set CUDA properties for the kernel library
set_property(TARGET kernel PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET kernel PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link CUDA runtime for kernel library
target_link_libraries(kernel ${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES})
