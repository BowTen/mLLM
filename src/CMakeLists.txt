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
)
add_library(kernel
    kernel/kernel.cpp
    kernel/embedding_kernel.cpp
)
add_library(model 
    model/qwen3.cpp
)

# Link libraries
target_link_libraries(base glog)
target_link_libraries(tokenizer base)
target_link_libraries(kernel base)
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

# Optional: Find and link CUDA if available
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_link_libraries(base ${CUDA_LIBRARIES})
    target_include_directories(base PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
