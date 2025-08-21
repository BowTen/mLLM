cmake_minimum_required(VERSION 3.14)

# Find Google Test from system installation
set(CMAKE_PREFIX_PATH /usr/local ${CMAKE_PREFIX_PATH})
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTEST REQUIRED gtest)
pkg_check_modules(GTEST_MAIN REQUIRED gtest_main)
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Create test executable
add_executable(tokenizer_test tokenizer_test.cpp)
add_executable(allocator_test allocator_test.cpp)
add_executable(buffer_test buffer_test.cpp)
add_executable(tensor_test tensor_test.cpp)
add_executable(safetensors_test safetensors_test.cpp)
add_executable(emb_kernel_test emb_kernel_test.cpp)
add_executable(qwen3_test qwen3_test.cpp)
add_executable(rms_norm_kernel_test rms_nrom_kernel_test.cpp)
add_executable(mat_add_kernel_test mat_add_kernel_test.cpp)
add_executable(mat_mul_kernel_test mat_mul_kernel_test.cpp)

# Link libraries
target_link_libraries(tokenizer_test 
    tokenizer  # Link to the tokenizer library from src
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(tokenizer_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(tokenizer_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(allocator_test 
    base  # Link to the base library from src
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(allocator_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(allocator_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(buffer_test 
    base  # Link to the base library from src
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(buffer_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(buffer_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(tensor_test 
    base  # Link to the base library from src
    kernel
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(tensor_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(tensor_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(safetensors_test 
    base  # Link to the base library from src
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(safetensors_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(safetensors_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(emb_kernel_test 
    base  # Link to the base library from src
    kernel  # Link to the kernel library from src
    model
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
    armadillo
    openblas
)
target_include_directories(emb_kernel_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(emb_kernel_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(qwen3_test 
    base  # Link to the base library from src
    kernel  # Link to the kernel library from src
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
    model
    glog
    armadillo
    openblas
)
target_include_directories(qwen3_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(qwen3_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(rms_norm_kernel_test 
    base  # Link to the base library from src
    kernel  # Link to the kernel library from src
    model
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
    armadillo
    openblas
)
target_include_directories(rms_norm_kernel_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(rms_norm_kernel_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(mat_add_kernel_test 
    base  # Link to the base library from src
    kernel  # Link to the kernel library from src
    op
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
    armadillo
    openblas
)
target_include_directories(mat_add_kernel_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(mat_add_kernel_test PRIVATE ${GTEST_CFLAGS_OTHER})

target_link_libraries(mat_mul_kernel_test 
    base  # Link to the base library from src
    kernel  # Link to the kernel library from src
    op
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
    armadillo
    openblas
)
target_include_directories(mat_mul_kernel_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(mat_mul_kernel_test PRIVATE ${GTEST_CFLAGS_OTHER})

# Enable testing
enable_testing()

# Add test
add_test(NAME TokenizerTest COMMAND tokenizer_test)
add_test(NAME AllocatorTest COMMAND allocator_test)
add_test(NAME BufferTest COMMAND buffer_test)
add_test(NAME TensorTest COMMAND tensor_test)
add_test(NAME SafeTensorsTest COMMAND safetensors_test)
add_test(NAME EmbKernelTest COMMAND emb_kernel_test)
add_test(NAME Qwen3Test COMMAND qwen3_test)
add_test(NAME RMSNormKernelTest COMMAND rms_norm_kernel_test)
add_test(NAME AddKernelTest COMMAND mat_add_kernel_test)
add_test(NAME MatMulKernelTest COMMAND mat_mul_kernel_test)
