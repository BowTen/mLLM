cmake_minimum_required(VERSION 3.14)

# Demo executables
add_executable(qwen3_chat qwen3_chat.cpp)

# Link required libraries (reuse those defined in src/CMakeLists)
target_link_libraries(qwen3_chat
    PRIVATE
        model
        op
        kernel
        tokenizer
        base
        glog
        armadillo
        openblas
)

# Include public headers
target_include_directories(qwen3_chat PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Debug flags for demo if in Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(qwen3_chat PRIVATE -g -O0)
    target_compile_definitions(qwen3_chat PRIVATE DEBUG)
endif()
