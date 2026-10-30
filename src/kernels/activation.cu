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

// Backward through silu(gate) * up.
// d_swiglu: upstream gradient [n]
// gate:     original gate values  [n]   (pre-activation)
// up:       original up values    [n]
// d_gate:   output gradient for gate  [n]
// d_up:     output gradient for up    [n]
__global__ void silu_backward_kernel(const float* d_swiglu, const float* gate,
                                     const float* up, float* d_gate, float* d_up,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    float silu_g = g * sig;
    // d(silu(g))/dg = sig * (1 + g * (1 - sig))
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    d_gate[i] = d_swiglu[i] * up[i] * dsilu;
    d_up[i]   = d_swiglu[i] * silu_g;
}

void silu_backward(const float* d_swiglu, const float* gate, const float* up,
                   float* d_gate, float* d_up, int n) {
    silu_backward_kernel<<<CEIL_DIV(n, 256), 256>>>(d_swiglu, gate, up, d_gate, d_up, n);
    CUDA_CHECK_LAST();
}