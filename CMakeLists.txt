cmake_minimum_required(VERSION 3.18)
project(cuda-llm-engine LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# gpu architecture (sm_80+ required for WMMA tensor cores)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
endif()

# include headers 
include_directories(${PROJECT_SOURCE_DIR}/include)

# kernel sources
set(KERNEL_SOURCES
    src/kernels/embed.cu
    src/kernels/matmul.cu
    src/kernels/matmul_wmma.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/softmax.cu
    src/kernels/activation.cu
    src/kernels/elementwise.cu
    src/kernels/attention.cu
    src/kernels/flash_attention.cu
    src/kernels/mlp.cu
    src/kernels/forward.cu
)

# main executable
add_executable(llm_engine src/main.cu ${KERNEL_SOURCES})

# wmma needs explicit arch flag
set_source_files_properties(src/kernels/matmul_wmma.cu PROPERTIES
    COMPILE_FLAGS "--generate-code=arch=compute_80,code=sm_80"
)