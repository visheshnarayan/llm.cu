#include <cuda_fp16.h>
#include "tensor.h"
#include "config.h"
#include <float.h>

void matmul_wmma_forward(const float* A, const half* B_half, float* C, int M, int N, int K);
void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols);
void rope_forward_decode(float* q, float* k, int pos_offset);

// single-query attention: one query position against cache_len key positions
// Q: [1, HEAD_DIM], K_cache: [cache_len, HEAD_DIM], V_cache: [cache_len, HEAD_DIM]
// out: [1, HEAD_DIM]
// one block per head, each thread handles part of the dot products
__global__ void decode_attn_kernel(
    const float* Q,          // [NUM_HEADS, HEAD_DIM]
    const float* K_cache,    // [cache_len, NUM_KV_HEADS, HEAD_DIM]
    const float* V_cache,    // [cache_len, NUM_KV_HEADS, HEAD_DIM]
    float* out,              // [NUM_HEADS, HEAD_DIM]
    int cache_len,
    float scale
) {
    int head = blockIdx.x;
    int tid = threadIdx.x;
    int kv_head = head / KV_HEAD_GROUPS;

    // Q for this head
    const float* q_head = Q + head * HEAD_DIM;

    // shared memory for scores
    extern __shared__ float smem[];  // [cache_len]

    // step 1: compute scores = Q @ K^T for all cached positions
    // each thread handles a subset of positions
    float max_score = -FLT_MAX;
    for (int pos = tid; pos < cache_len; pos += blockDim.x) {
        const float* k_pos = K_cache + pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            dot += q_head[d] * k_pos[d];
        }
        dot *= scale;
        smem[pos] = dot;
        if (dot > max_score) max_score = dot;
    }
    __syncthreads();

    // step 2: find global max (tree reduction)
    __shared__ float smax[256];
    smax[tid] = max_score;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }
    float global_max = smax[0];

    // step 3: exp and sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < cache_len; pos += blockDim.x) {
        float val = expf(smem[pos] - global_max);
        smem[pos] = val;
        local_sum += val;
    }
    __shared__ float ssum[256];
    ssum[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) ssum[tid] += ssum[tid + s];
        __syncthreads();
    }
    float total_sum = ssum[0];

    // step 4: weighted sum of V
    // each thread accumulates partial result across positions
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    for (int pos = tid; pos < cache_len; pos += blockDim.x) {
        float w = smem[pos] / total_sum;
        const float* v_pos = V_cache + pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] += w * v_pos[d];
        }
    }

    // reduce across threads for each dimension
    // use shared memory to sum partial results
    for (int d = 0; d < HEAD_DIM; d++) {
        ssum[tid] = acc[d];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) ssum[tid] += ssum[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            out[head * HEAD_DIM + d] = ssum[0];
        }
        __syncthreads();
    }
}

// append new K/V to cache at position cache_len
// new_k: [1, NUM_KV_HEADS * HEAD_DIM]
// k_cache: [MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM]
__global__ void append_kv_kernel(
    const float* new_kv, float* cache,
    int cache_pos, int kv_dim
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kv_dim) {
        cache[cache_pos * kv_dim + i] = new_kv[i];
    }
}

// single-token decode attention
// hidden: [1, HIDDEN_SIZE] (one new token)
// k_cache, v_cache: [MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM]
// cache_len: how many positions already cached (new token goes at cache_len)
void attention_decode(
    const float* hidden,
    const half* wq_h, const half* wk_h, const half* wv_h, const half* wo_h,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out,
    float* q, float* k, float* v,   // workspace [1, dim]
    float* attn_out,                 // workspace [1, NUM_HEADS * HEAD_DIM]
    float* k_cache, float* v_cache,  // per-layer cache
    int cache_len                     // current cache length (before this token)
) {
    int q_dim = NUM_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    // project Q, K, V for the new token (seq_len=1)
    matmul_wmma_forward(hidden, wq_h, q, 1, q_dim, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, wk_h, k, 1, kv_dim, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, wv_h, v, 1, kv_dim, HIDDEN_SIZE);

    // QK-norm
    rmsnorm_forward(q, q_norm_weight, q, NUM_HEADS, HEAD_DIM);
    rmsnorm_forward(k, k_norm_weight, k, NUM_KV_HEADS, HEAD_DIM);

    // RoPE at absolute position cache_len
    rope_forward_decode(q, k, cache_len);

    // append new K, V to cache
    int block = 256;
    int grid = CEIL_DIV(kv_dim, block);
    append_kv_kernel<<<grid, block>>>(k, k_cache, cache_len, kv_dim);
    CUDA_CHECK_LAST();
    append_kv_kernel<<<grid, block>>>(v, v_cache, cache_len, kv_dim);
    CUDA_CHECK_LAST();

    int total_len = cache_len + 1;  // including the new token

    // decode attention: Q[1] against all cached K/V
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    size_t smem_size = total_len * sizeof(float);
    decode_attn_kernel<<<NUM_HEADS, 256, smem_size>>>(
        q, k_cache, v_cache, attn_out, total_len, scale
    );
    CUDA_CHECK_LAST();

    // output projection
    matmul_wmma_forward(attn_out, wo_h, out, 1, HIDDEN_SIZE, q_dim);
}

void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales, float* C, int M, int N, int K);

void attention_decode_int8(
    const float* hidden,
    const int8_t* wq_q, const float* wq_s,
    const int8_t* wk_q, const float* wk_s,
    const int8_t* wv_q, const float* wv_s,
    const int8_t* wo_q, const float* wo_s,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out,
    float* q, float* k, float* v,
    float* attn_out,
    float* k_cache, float* v_cache,
    int cache_len
) {
    int q_dim = NUM_HEADS * HEAD_DIM;
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;

    matmul_int8_forward(hidden, wq_q, wq_s, q, 1, q_dim, HIDDEN_SIZE);
    matmul_int8_forward(hidden, wk_q, wk_s, k, 1, kv_dim, HIDDEN_SIZE);
    matmul_int8_forward(hidden, wv_q, wv_s, v, 1, kv_dim, HIDDEN_SIZE);

    rmsnorm_forward(q, q_norm_weight, q, NUM_HEADS, HEAD_DIM);
    rmsnorm_forward(k, k_norm_weight, k, NUM_KV_HEADS, HEAD_DIM);

    rope_forward_decode(q, k, cache_len);

    int block = 256;
    int grid = CEIL_DIV(kv_dim, block);
    append_kv_kernel<<<grid, block>>>(k, k_cache, cache_len, kv_dim);
    CUDA_CHECK_LAST();
    append_kv_kernel<<<grid, block>>>(v, v_cache, cache_len, kv_dim);
    CUDA_CHECK_LAST();

    int total_len = cache_len + 1;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    size_t smem_size = total_len * sizeof(float);
    decode_attn_kernel<<<NUM_HEADS, 256, smem_size>>>(
        q, k_cache, v_cache, attn_out, total_len, scale
    );
    CUDA_CHECK_LAST();

    matmul_int8_forward(attn_out, wo_q, wo_s, out, 1, HIDDEN_SIZE, q_dim);
}
