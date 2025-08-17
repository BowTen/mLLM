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

# Link libraries
target_link_libraries(tokenizer_test 
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    Threads::Threads
)
target_include_directories(tokenizer_test PRIVATE ${GTEST_INCLUDE_DIRS})
target_compile_options(tokenizer_test PRIVATE ${GTEST_CFLAGS_OTHER})

# Enable testing
enable_testing()

# Add test
add_test(NAME TokenizerTest COMMAND tokenizer_test)
