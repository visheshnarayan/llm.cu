#ifndef UTILS_H
#define UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// wrap every cuda api call with this to catch error instead of corrupting future ops
#define CUDA_CHECK(err) do { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// after kernel launch. kernels are async so errors dont surface until explicit check
#define CUDA_CHECK_LAST() do { \
    cudaError_t err_ = cudaGetLastError(); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// ceiling div for grid dims
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

#endif // UTILS_H
