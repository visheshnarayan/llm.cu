#include "tensor.h"

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// one thread per element
__global__ void silu_kernel(const float* gate, const float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { 
        float g = gate[i]; 
        out[i] = (g / (1.0f + expf(-g))) * up[i]; 
    }
}

void silu_forward(const float* gate, const float* up, float* out, int n) { 
    dim3 block(256); 
    dim3 grid(CEIL_DIV(n, 256));
    silu_kernel<<<grid, block>>>(gate, up, out, n);
    CUDA_CHECK_LAST();
}