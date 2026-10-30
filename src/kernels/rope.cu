#include "tensor.h"
#include "config.h"

// grid=(seq_len, num_heads), block=(HEAD_DIM/2)
// each thread rotates one pair (x[d], x[d + HEAD_DIM/2]) :: half-split convention
// HF Qwen3 uses rotate_half: pairs (0,64), (1,65), ..., (63,127)
__global__ void rope_kernel(float* q, float* k, int seq_len, int num_q_heads, int num_kv_heads, int pos_offset) {
    int local_pos = blockIdx.x;                    // index into q/k arrays
    int abs_pos = local_pos + pos_offset;          // absolute position for angle
    int head = blockIdx.y;
    int d = threadIdx.x;  // dimension in [0, HEAD_DIM/2)
    int half = HEAD_DIM / 2;

    // compute rotation angle using absolute position
    float freq = 1.0f / powf(ROPE_THETA, (2.0f * d) / HEAD_DIM);
    float angle = abs_pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // rotate Q: pair (d, d + half)
    if (head < num_q_heads) {
        int base = local_pos * num_q_heads * HEAD_DIM + head * HEAD_DIM;
        float x0 = q[base + d];
        float x1 = q[base + d + half];
        q[base + d]        = x0 * cos_val - x1 * sin_val;
        q[base + d + half] = x1 * cos_val + x0 * sin_val;
    }

    // rotate K (only for KV heads)
    if (head < num_kv_heads) {
        int base = local_pos * num_kv_heads * HEAD_DIM + head * HEAD_DIM;
        float x0 = k[base + d];
        float x1 = k[base + d + half];
        k[base + d]        = x0 * cos_val - x1 * sin_val;
        k[base + d + half] = x1 * cos_val + x0 * sin_val;
    }
}

// prefill: positions 0..seq_len-1
void rope_forward(float* q, float* k, int seq_len) {
    dim3 grid(seq_len, NUM_HEADS);
    dim3 block(HEAD_DIM / 2);
    rope_kernel<<<grid, block>>>(q, k, seq_len, NUM_HEADS, NUM_KV_HEADS, 0);
    CUDA_CHECK_LAST();
}

// decode: single token at absolute position pos_offset
void rope_forward_decode(float* q, float* k, int pos_offset) {
    dim3 grid(1, NUM_HEADS);
    dim3 block(HEAD_DIM / 2);
    rope_kernel<<<grid, block>>>(q, k, 1, NUM_HEADS, NUM_KV_HEADS, pos_offset);
    CUDA_CHECK_LAST();
}

// Backward through RoPE (orthogonal transform → transpose = inverse rotation).
// Applies (cos, +sin) instead of (cos, -sin) to invert the rotation.
__global__ void rope_backward_kernel(float* dq, float* dk, int seq_len,
                                     int num_q_heads, int num_kv_heads) {
    int local_pos = blockIdx.x;
    int head      = blockIdx.y;
    int d         = threadIdx.x;
    int half      = HEAD_DIM / 2;

    float freq  = 1.0f / powf(ROPE_THETA, (2.0f * d) / HEAD_DIM);
    float angle = local_pos * freq;
    float cos_val =  cosf(angle);
    float sin_val =  sinf(angle);   // +sin for inverse

    if (head < num_q_heads) {
        int base = local_pos * num_q_heads * HEAD_DIM + head * HEAD_DIM;
        float dy0 = dq[base + d];
        float dy1 = dq[base + d + half];
        dq[base + d]        = dy0 * cos_val + dy1 * sin_val;
        dq[base + d + half] = dy1 * cos_val - dy0 * sin_val;
    }
    if (head < num_kv_heads) {
        int base = local_pos * num_kv_heads * HEAD_DIM + head * HEAD_DIM;
        float dy0 = dk[base + d];
        float dy1 = dk[base + d + half];
        dk[base + d]        = dy0 * cos_val + dy1 * sin_val;
        dk[base + d + half] = dy1 * cos_val - dy0 * sin_val;
    }
}

void rope_backward(float* dq, float* dk, int seq_len) {
    dim3 grid(seq_len, NUM_HEADS);
    dim3 block(HEAD_DIM / 2);
    rope_backward_kernel<<<grid, block>>>(dq, dk, seq_len, NUM_HEADS, NUM_KV_HEADS);
    CUDA_CHECK_LAST();
}