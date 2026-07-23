#include "tensor.h"

// residual: out = x + residual 
__global__ void residual_kernel(const float* x, const float* residual, float* out, int n) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { 
        out[i] = x[i] + residual[i]; 
    }
}

void residual_forward(const float* x, const float* residual, float* out, int n) {
    dim3 block(256); 
    dim3 grid(CEIL_DIV(n, 256)); 
    residual_kernel<<<grid, block>>>(x, residual, out, n);
    CUDA_CHECK_LAST();
}