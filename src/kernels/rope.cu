#include "tensor.h"
#include "config.h"

// grid=(seq_len, num_heads), block=(HEAD_DIM/2)
// each thread rotates one pair (x[d], x[d + HEAD_DIM/2]) :: half-split convention
// HF Qwen3 uses rotate_half: pairs (0,64), (1,65), ..., (63,127)
__global__ void rope_kernel(float* q, float* k, int seq_len, int num_q_heads, int num_kv_heads) {
    int pos = blockIdx.x;
    int head = blockIdx.y;
    int d = threadIdx.x;  // dimension in [0, HEAD_DIM/2)
    int half = HEAD_DIM / 2;

    // compute rotation angle
    float freq = 1.0f / powf(ROPE_THETA, (2.0f * d) / HEAD_DIM);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // rotate Q: pair (d, d + half)
    if (head < num_q_heads) {
        int base = pos * num_q_heads * HEAD_DIM + head * HEAD_DIM;
        float x0 = q[base + d];
        float x1 = q[base + d + half];
        q[base + d]        = x0 * cos_val - x1 * sin_val;
        q[base + d + half] = x1 * cos_val + x0 * sin_val;
    }

    // rotate K (only for KV heads)
    if (head < num_kv_heads) {
        int base = pos * num_kv_heads * HEAD_DIM + head * HEAD_DIM;
        float x0 = k[base + d];
        float x1 = k[base + d + half];
        k[base + d]        = x0 * cos_val - x1 * sin_val;
        k[base + d + half] = x1 * cos_val + x0 * sin_val;
    }
}

void rope_forward(float* q, float* k, int seq_len) {
    dim3 grid(seq_len, NUM_HEADS); // NUM_HEADS >= NUM_KV_HEADS
    dim3 block(HEAD_DIM / 2); 
    rope_kernel<<<grid, block>>>(q, k, seq_len, NUM_HEADS, NUM_KV_HEADS); 
    CUDA_CHECK_LAST();
}