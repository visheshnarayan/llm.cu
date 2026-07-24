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

    // per-head attention: scores = Q @ K^T , mask + scale, softmax, attn_out = scores @ V
    // scaling query info onto the key (how well does this question align with this context)
    // take attention as scaling on the info we extract from the value subspace :: decodes to next word
    for (int h = 0; h < NUM_HEADS; h++) { // stride: q_dim per row
        float* q_h = q + h * HEAD_DIM; 
        float* k_h = k_exp + h * HEAD_DIM; 
        float* v_h = v_exp + h * HEAD_DIM; 
        float* s_h = scores + h * seq_len * seq_len;
        float* o_h = attn_out + h * HEAD_DIM; 

        // scores[seq,seq] = Q_h[seq,HEAD_DIM] @ K_h^T[HEAD_DIM,seq]
        // TODO: optimize with batched matmul
        matmul_forward(q_h, k_h, s_h, seq_len, seq_len, HEAD_DIM); 

        // casual mask + scale 
        mask_scale_kernel<<<seq_len, 256>>>(s_h, seq_len);
        CUDA_CHECK_LAST();

        // softmax each row 
        softmax_forward(s_h, seq_len, seq_len);

        // attn_out[seq, HEAD_DIM] = scores[seq, seq] @ V_h[seq, HEAD_DIM]
        matmul_forward(s_h, v_h, o_h, seq_len, HEAD_DIM, seq_len);
    }

    // output projection: out[seq, HIDDEN_SIZE] = attn_out[seq, q_dim] @ Wo^T
    matmul_forward(attn_out, wo, out, seq_len, HIDDEN_SIZE, q_dim); 
}