cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Project Name
project(GPT-2 CXX)

# Directory Name
set(SUB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../utils)
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Create Executable File
set(SRCS
    ${SRC_DIR}/main.cpp
)

add_subdirectory(${SUB_DIR} build)
