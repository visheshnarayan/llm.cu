#include "tensor.h"
#include "config.h"

// grid=(seq_len, num_heads), block=(HEAD_DIM/2)
// each thread rotates one pair (x[2i], x[2i+1])
__global__ void rope_kernel(float* q, float* k, int seq_len, int num_q_heads, int num_kv_heads) {
    int pos = blockIdx.x; 
    int head = blockIdx.y;
    int pair = threadIdx.x;  // which pair in [0, HEAD_DIM/2]

    // compute rotation angle 
    float freq = 1.0f / powf(ROPE_THETA, (2.0f * pair) / HEAD_DIM); 
    float angle = pos * freq; 
    float cos_val = cosf(angle); 
    float sin_val = sinf(angle); 

    // rotate Q
    if (head < num_q_heads) {
        int idx = pos * num_q_heads * HEAD_DIM + head * HEAD_DIM + 2 * pair; 
        float x0 = q[idx]; 
        float x1 = q[idx + 1]; 
        q[idx] = x0 * cos_val - x1 * sin_val; 
        q[idx + 1] = x0 * sin_val + x1 * cos_val;
    }

    // rotate K (only for KV heads) 
    if (head < num_kv_heads) {
        int idx = pos * num_kv_heads * HEAD_DIM + head * HEAD_DIM + 2 * pair; 
        float x0 = k[idx]; 
        float x1 = k[idx + 1];
        k[idx] = x0 * cos_val - x1 * sin_val; 
        k[idx + 1] = x0 * sin_val + x1 * cos_val; 
    
}

void rope_forward(float* q, float* k, int seq_len) {
    dim3 grid(seq_len, NUM_HEADS); // NUM_HEADS >= NUM_KV_HEADS
    dim3 block(HEAD_DIM / 2); 
    rope_kernel<<<grid, block>>>(q, k, seq_len, NUM_HEADS, NUM_KV_HEADS); 
    CUDA_CHECK_LAST();
}