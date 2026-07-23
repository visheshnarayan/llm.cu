#include "tensor.h"

// one block per row 256 threads 
// three-pass: find max, compute exp(x - max), normalize by sum 
__global__ void softmax_kernel(float* x, int rows, int cols) {
    int row = blockIdx.x; 
    float* x_row = x + row * cols; 

    __shared__ float shared[256]; 

    // pass 1 find max 
    float max_val = -1e30f;
    for ( int i = threadIdx.x ; i < cols ; i += blockDim.x ) { 
        max_val = fmaxf(max_val, x_row[i]);
    }
    shared[threadIdx.x] = max_val; 
    __syncthreads();

    for ( int stride = blockDim.x / 2 ; stride > 0 ; stride >>= 1 ) { 
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = shared[0]; 
    __syncthreads();

    // pass 2 exp(x - max) and sum
    float sum = 0.0f; 
    for ( int i = threadIdx.x ; i < cols ; i += blockDim.x ) { 
        x_row[i] = expf(x_row[i] - max_val); 
        sum += x_row[i]; 
    }
    shared[threadIdx.x] = sum; 
    __syncthreads(); 

    for ( int stride = blockDim.x / 2 ; stride > 0 ; stride >>= 1 ) { 
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride]; 
        }
        __syncthreads();
    }
    sum = shared[0]; 
    __syncthreads(); 

    // pass 3 normalize 
    for ( int i = threadIdx.x ; i < cols ; i += blockDim.x ) { 
        x_row[i] /= sum;
    }
}

void softmax_forward(float* x, int rows, int cols) { 
    dim3 grid(rows); 
    dim3 block(256); 
    softmax_kernel<<<grid, block>>>(x, rows, cols); 
    CUDA_CHECK_LAST();
}