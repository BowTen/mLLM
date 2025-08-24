cmake_minimum_required(VERSION 3.14)
project(MiniLLM VERSION 1.0 LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Debug flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")
    message(STATUS "Debug mode enabled")
endif()

# Add compile options
add_compile_options(-Wall -Wextra)

# Include directories
include_directories(include)

# Add subdirectories
add_subdirectory(src)
add_subdirectory(test)
