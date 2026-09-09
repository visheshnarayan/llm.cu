#include "tensor.h"
#include "config.h"
#include <math.h> 

// forward declarations of kernels we call
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_wmma_forward(const float* A, const half* B_half, float* C, int M, int N, int K);
void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols);
void rope_forward(float* q, float* k, int seq_len);
void flash_attention_forward(const float* Q, const float* K, const float* V, float* O, int seq_len); 

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
    const half* wq_h,               // [NUM_HEADS * HEAD_DIM, HIDDEN_SIZE] fp16
    const half* wk_h,               // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE] fp16
    const half* wv_h,               // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE] fp16
    const half* wo_h,               // [HIDDEN_SIZE, NUM_HEADS * HEAD_DIM] fp16
    const float* q_norm_weight,     // [HEAD_DIM]
    const float* k_norm_weight,     // [HEAD_DIM]
    float* out,                     // [seq, HIDDEN_SIZE]
    float* q, float* k, float* v,   // workspace
    float* k_exp, float* v_exp,     // expanded KV
    float* attn_out,                // workspace [seq, NUM_HEADS * HEAD_DIM]
    float* q_head, float* k_head,   // per-head workspace [seq, HEAD_DIM] each
    float* v_head, float* o_head,   // per-head workspace [seq, HEAD_DIM] each
    int seq_len
) {
    int q_dim = NUM_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // q, k, v projections (tensor cores, fp16 weights)
    matmul_wmma_forward(hidden, wq_h, q, seq_len, q_dim, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, wk_h, k, seq_len, kv_dim, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, wv_h, v, seq_len, kv_dim, HIDDEN_SIZE);

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

    // per-head flash attention: extract head -> fused QK^T+mask+softmax+V -> scatter back
    for (int h = 0; h < NUM_HEADS; h++) {
        // extract this head's q, k, v into contiguous [seq, HEAD_DIM] buffers
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(q, q_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(k_exp, k_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(v_exp, v_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();

        // fused attention: o_head = softmax(Q@K^T / sqrt(d) + mask) @ V
        flash_attention_forward(q_head, k_head, v_head, o_head, seq_len);

        // scatter back into interleaved attn_out
        scatter_head_kernel<<<seq_len, HEAD_DIM>>>(o_head, attn_out, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
    }

    // output projection: out[seq, HIDDEN_SIZE] = attn_out[seq, q_dim] @ Wo^T
    matmul_wmma_forward(attn_out, wo_h, out, seq_len, HIDDEN_SIZE, q_dim);
}

void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales, float* C, int M, int N, int K);

void attention_forward_int8(
    const float* hidden,
    const int8_t* wq_q, const float* wq_s,
    const int8_t* wk_q, const float* wk_s,
    const int8_t* wv_q, const float* wv_s,
    const int8_t* wo_q, const float* wo_s,
    const float* q_norm_weight,
    const float* k_norm_weight,
    float* out,
    float* q, float* k, float* v,
    float* k_exp, float* v_exp,
    float* attn_out,
    float* q_head, float* k_head,
    float* v_head, float* o_head,
    int seq_len
) {
    int q_dim = NUM_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    matmul_int8_forward(hidden, wq_q, wq_s, q, seq_len, q_dim, HIDDEN_SIZE);
    matmul_int8_forward(hidden, wk_q, wk_s, k, seq_len, kv_dim, HIDDEN_SIZE);
    matmul_int8_forward(hidden, wv_q, wv_s, v, seq_len, kv_dim, HIDDEN_SIZE);

    rmsnorm_forward(q, q_norm_weight, q, seq_len * NUM_HEADS, HEAD_DIM);
    rmsnorm_forward(k, k_norm_weight, k, seq_len * NUM_KV_HEADS, HEAD_DIM);

    rope_forward(q, k, seq_len);

    dim3 exp_grid(seq_len, NUM_HEADS);
    dim3 exp_block(HEAD_DIM);
    expand_kv_kernel<<<exp_grid, exp_block>>>(k, k_exp, seq_len);
    CUDA_CHECK_LAST();
    expand_kv_kernel<<<exp_grid, exp_block>>>(v, v_exp, seq_len);
    CUDA_CHECK_LAST();

    for (int h = 0; h < NUM_HEADS; h++) {
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(q, q_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(k_exp, k_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
        extract_head_kernel<<<seq_len, HEAD_DIM>>>(v_exp, v_head, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();

        flash_attention_forward(q_head, k_head, v_head, o_head, seq_len);

        scatter_head_kernel<<<seq_len, HEAD_DIM>>>(o_head, attn_out, seq_len, NUM_HEADS, h);
        CUDA_CHECK_LAST();
    }

    matmul_int8_forward(attn_out, wo_q, wo_s, out, seq_len, HIDDEN_SIZE, q_dim);
}