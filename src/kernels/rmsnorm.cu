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

// Backward through RMSNorm.
// d_out: [rows, cols]  upstream gradient
// x_in:  [rows, cols]  original pre-norm input
// weight: [cols]
// d_x:   [rows, cols]  output gradient (written, not accumulated)
//
// Math (per row):
//   r    = rsqrt(mean(x^2) + eps)
//   c    = r^2/cols * dot(d_out * weight, x_in)
//   d_x  = r * (d_out * weight  -  c * x_in)
__global__ void rmsnorm_backward_kernel(const float* d_out, const float* x_in,
                                         const float* weight, float* d_x,
                                         int cols) {
    int row = blockIdx.x;
    const float* do_row = d_out  + row * cols;
    const float* xi_row = x_in   + row * cols;
    float*       dx_row = d_x    + row * cols;

    __shared__ float shared[256];

    // 1) recompute rms for this row
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        ss += xi_row[i] * xi_row[i];
    shared[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float r = rsqrtf(shared[0] / cols + RMS_NORM_EPS);

    // 2) c = (r^2 / cols) * dot(d_out * weight, x_in)
    float dot = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dot += do_row[i] * weight[i] * xi_row[i];
    shared[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float c = (r * r / cols) * shared[0];

    // 3) d_x = r * (d_out * weight - c * x_in)
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        dx_row[i] = r * (do_row[i] * weight[i] - c * xi_row[i]);
}

void rmsnorm_backward(const float* d_out, const float* x_in, const float* weight,
                      float* d_x, int rows, int cols) {
    rmsnorm_backward_kernel<<<rows, 256>>>(d_out, x_in, weight, d_x, cols);
    CUDA_CHECK_LAST();
}