cmake_minimum_required(VERSION 3.14)

# Create a library for the tokenizer
add_library(tokenizer INTERFACE)
target_include_directories(tokenizer INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/base
)
