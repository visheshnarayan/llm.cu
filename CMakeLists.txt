cmake_minimum_required(VERSION 3.18)
project(cuda-inference-lora LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# if gpu architecture, fall back to common ones 
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES) 
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# include headers 
include_directories(${PROJECT_SOURCE_DIR}/include)

# TODO:  add_executable and kernel sources 