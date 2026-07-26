#include "tensor.h"
#include "config.h"
#include <math.h> 

// forward declarations of kernels we call
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K); 
void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols); 
void rope_forward(float* q, float* k, int seq_len); 
void softmax_forward(float* x, int rows, int cols); 
void residual_forward(const float* x, const float* residual, float* out, int n); 

// casual mask + scale: scors[i][j] = (j > i) ? -inf : scores[i][j] / sqrt(HEAD_DIM)
__global__ void mask_scale_kernel(float* scores, int seq_len) {
    int row = blockIdx.x;   // query pos
    int col = threadIdx.x;  // key pos 

    for (int j = col; j < seq_len; j += blockDim.x) {
        int idx = row * seq_len + j; 
        if (j > row) {
            scores[idx] = -1e30f;
        } else {
            scores[idx] /= sqrtf((float)HEAD_DIM); 
        }
    }
}

// expand KV heads for GQA: copy each KV head to its grouped Q head
// in: [seq, NUM_KV_HEADS, HEAD_DIM] -> out: [seq, NUM_HEADS, HEAD_DIM]
__global__ void expand_kv_kernel(const float* kv_in, float* kv_out, int seq_len) {
    int pos = blockIdx.x; 
    int head = blockIdx.y; 
    int d = threadIdx.x; 

    int kv_head = head / KV_HEAD_GROUPS; 
    int in_idx = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d; 
    int out_idx = pos * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + d; 

    if (d < HEAD_DIM) {
        kv_out[out_idx] = kv_in[in_idx];
    }
}

// extract one head from interleaved [seq, num_heads, HEAD_DIM] into contiguous [seq, HEAD_DIM]
__global__ void extract_head_kernel(const float* in, float* out, int seq_len, int num_heads, int head) {
    int pos = blockIdx.x;
    int d = threadIdx.x;
    if (pos < seq_len && d < HEAD_DIM) {
        out[pos * HEAD_DIM + d] = in[pos * num_heads * HEAD_DIM + head * HEAD_DIM + d];
    }
}

// scatter one head from contiguous [seq, HEAD_DIM] back into interleaved [seq, num_heads, HEAD_DIM]
__global__ void scatter_head_kernel(const float* in, float* out, int seq_len, int num_heads, int head) {
    int pos = blockIdx.x;
    int d = threadIdx.x;
    if (pos < seq_len && d < HEAD_DIM) {
        out[pos * num_heads * HEAD_DIM + head * HEAD_DIM + d] = in[pos * HEAD_DIM + d];
    }
}

void attention_forward(
    const float* hidden,            // [seq, HIDDEN_SIZE]
    const float* wq,                // [NUM_HEADS * HEAD_DIM, HIDDEN_SIZE]
    const float* wk,                // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE]
    const float* wv,                // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE]
    const float* wo,                // [HIDDEN_SIZE, NUM_HEADS * HEAD_DIM]
    const float* q_norm_weight,     // [HEAD_DIM]
    const float* k_norm_weight,     // [HEAD_DIM]
    float* out,                     // [seq, HIDDEN_SIZE]
    float* q, float* k, float* v,   // workspace
    float* k_exp, float* v_exp,     // expanded KV
    float* scores, float* attn_out, // workspace
    float* q_head, float* k_head,   // per-head workspace [seq, HEAD_DIM] each
    float* v_head, float* o_head,   // per-head workspace [seq, HEAD_DIM] each
    int seq_len
) {
    int q_dim = NUM_HEADS * HEAD_DIM; 
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // q, k, v projections 
    matmul_forward(hidden, wq, q, seq_len, q_dim, HIDDEN_SIZE);
    matmul_forward(hidden, wk, k, seq_len, kv_dim, HIDDEN_SIZE);
    matmul_forward(hidden, wv, v, seq_len, kv_dim, HIDDEN_SIZE);

    // QK-norm (per head RMSNorm)
    rmsnorm_forward(q, q_norm_weight, q, seq_len * NUM_HEADS, HEAD_DIM); 
    rmsnorm_forward(k, k_norm_weight, k, seq_len * NUM_KV_HEADS, HEAD_DIM);

    // rope 
    rope_forward(q, k, seq_len);
    

    // expand KV for GQA
    dim3 exp_grid(seq_len, NUM_HEADS); 
    dim3 exp_block(HEAD_DIM); 
    expand_kv_kernel<<<exp_grid, exp_block>>>(k, k_exp, seq_len);
    CUDA_CHECK_LAST();
    expand_kv_kernel<<<exp_grid, exp_block>>>(v, v_exp, seq_len);
    CUDA_CHECK_LAST();

    // per-head attention: extract head -> Q@K^T -> mask+scale -> softmax -> scores@V -> scatter back
    for (int h = 0; h < NUM_HEADS; h++) {
        float* s_h = scores + h * seq_len * seq_len;

        // extract this head's q, k, v into contiguous [seq, HEAD_DIM] buffers
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(q, q_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(k_exp, k_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(v_exp, v_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();

        // scores[seq,seq] = Q_h[seq,HEAD_DIM] @ K_h^T[HEAD_DIM,seq]
        matmul_forward(q_head, k_head, s_h, seq_len, seq_len, HEAD_DIM);

        // causal mask + scale
        mask_scale_kernel<<<seq_len, 256>>>(s_h, seq_len);
        CUDA_CHECK_LAST();

        // softmax each row
        softmax_forward(s_h, seq_len, seq_len);

        // o_head[seq, HEAD_DIM] = scores[seq, seq] @ V_h[seq, HEAD_DIM]
        matmul_forward(s_h, v_head, o_head, seq_len, HEAD_DIM, seq_len);

        // scatter back into interleaved attn_out
        scatter_head_kernel<<<seq_len, HEAD_DIM>>>(o_head, attn_out, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
    }

    // output projection: out[seq, HIDDEN_SIZE] = attn_out[seq, q_dim] @ Wo^T
    matmul_forward(attn_out, wo, out, seq_len, HIDDEN_SIZE, q_dim); 
}