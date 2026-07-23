#include "tensor.h" 
#include "config.h"

// one block per row, 256 threads
// computes: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int cols) {
    int row = blockIdx.x; 
    const float* x_row = x + row * cols; 
    float* out_row = out + row * cols; 

    // each thread accumulates partial sum of squares
    __shared__ float shared[256]; 
    float sum_sq = 0.0f; 
    for ( int i = threadIdx.x ; i < cols ; i += blockDim.x ) {
        float val = x_row[i]; 
        sum_sq += val * val; 
    }
    shared[threadIdx.x] = sum_sq;
    __syncthreads();

    // tree reduction 
    for (int stride = blockDim.x / 2 ; stride > 0 ; stride >>= 1) { 
        if (threadIdx.x < stride) { 
            shared[threadIdx.x] += shared[threadIdx.x + stride]; 
        }
        __syncthreads(); 
    }

    // normalize scale 
    float rms = rsqrtf(shared[0] / cols + RMS_NORM_EPS); 
    for ( int i = threadIdx.x ; i < cols ; i += blockDim.x ) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}

void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols) { 
    dim3 grid(rows); 
    dim3 block(256); 
    rmsnorm_kernel<<<grid, block>>>(x, weight, out, cols); 
    CUDA_CHECK_LAST();
}