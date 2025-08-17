cmake_minimum_required(VERSION 3.14)

# Add base library
add_library(base 
    base/allocator.cpp
    base/buffer.cpp
)

# Add tokenizer library
add_library(tokenizer 
    tokenizer/tokenizer.cpp
)

# Link libraries
target_link_libraries(tokenizer base)

# Include directories for the libraries
target_include_directories(base PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(tokenizer PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

# Optional: Find and link CUDA if available
find_package(CUDA QUIET)
if(CUDA_FOUND)
    target_link_libraries(base ${CUDA_LIBRARIES})
    target_include_directories(base PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
