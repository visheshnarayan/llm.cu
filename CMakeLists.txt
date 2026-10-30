cmake_minimum_required(VERSION 3.18)
project(cuda-llm-engine LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# gpu architecture (sm_80+ required for WMMA tensor cores)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

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
    src/kernels/attention_decode.cu
    src/kernels/flash_attention.cu
    src/kernels/mlp.cu
    src/kernels/forward.cu
    src/kernels/quantize.cu
)

# main executable
add_executable(llm_engine src/main.cu ${KERNEL_SOURCES})

# matmul benchmark
add_executable(bench_matmul
    benchmarks/bench_matmul.cu
    src/kernels/matmul.cu
    src/kernels/matmul_wmma.cu
    src/kernels/quantize.cu
)

# LoRA fine-tuning trainer (M6)
set(LORA_SOURCES
    src/kernels/embed.cu
    src/kernels/matmul.cu
    src/kernels/rmsnorm.cu
    src/kernels/rope.cu
    src/kernels/activation.cu
    src/kernels/elementwise.cu
    src/kernels/lora.cu
    src/kernels/loss.cu
    src/kernels/lora_forward.cu
    src/kernels/lora_backward.cu
)
add_executable(llm_train src/train.cu ${LORA_SOURCES})
target_link_libraries(llm_train curand)
