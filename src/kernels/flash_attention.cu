#include "tensor.h"
#include "config.h"
#include <math.h>
#include <float.h>

// block size for tiles 
#define BLOCK_M 16  // tile rows :: query positions
#define BLOCK_N 16  // tile cols :: key positions

// flash attention kernel for a single head 
// Q, K, V: [seq_len, HEAD_DIM], O: [seq_len, HEAD_DIM]
// each thread block handles BLOCK_M rows of Q 
__global__ void flash_attn_kernel(
    const float* Q, const float* K, const float* V, float* O, 
    int seq_len, float scale
) {
    int q_start = blockIdx.x * BLOCK_M;
    int tid = threadIdx.x;

    // shared memory for tiles 
    __shared__ float q_tile[BLOCK_M][HEAD_DIM]; 
    __shared__ float k_tile[BLOCK_N][HEAD_DIM];
    __shared__ float v_tile[BLOCK_N][HEAD_DIM];
    __shared__ float scores[BLOCK_M][BLOCK_N];

    // per-row running stats for online softmax
    // each thread handles one row in the q tile 
    int local_row = tid; // one thread per row in BLOCK_M
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // accumulator for output: [BLOCK_M, HEAD_DIM] in registers
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; d++) {
        acc[d] = 0.0f;
    }

    // load q tile for output: [BLOCK_M, HEAD_DIM] in registers
    if (q_start + local_row < seq_len) {
        for (int d = 0; d < HEAD_DIM; d++) {
            q_tile[local_row][d] = Q[(q_start + local_row) * HEAD_DIM + d];
        }
    } else { // rest is 0 no tokens left in seq
        for (int d = 0; d < HEAD_DIM; d++) { 
            q_tile[local_row][d] = 0.0f;
        }
    }
    __syncthreads();

    // loop over KV blocks
    int num_kv_blocks = (seq_len + BLOCK_N - 1) / BLOCK_N; 
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * BLOCK_N; 

        // load K tile 
        if (kv_start + local_row < seq_len && local_row < BLOCK_N) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_tile[local_row][d] = K[(kv_start + local_row) * HEAD_DIM + d]; 
            }
        } else if (local_row < BLOCK_N) {
            for (int d = 0; d < HEAD_DIM; d++) {
                k_tile[local_row][d] = 0.0f;
            }
        }

        if (kv_start + local_row < seq_len && local_row < BLOCK_N) {
            for (int d = 0; d < HEAD_DIM; d++) {
                v_tile[local_row][d] = V[(kv_start + local_row) * HEAD_DIM + d]; 
            }
        } else if (local_row < BLOCK_N) {
            for (int d = 0; d < HEAD_DIM; d++) {
                v_tile[local_row][d] = 0.0f;
            }
        }
        __syncthreads();

        // compute scores: dot(q_tile[local_row] , k_tile[j]) for each j
        for (int j = 0; j < BLOCK_N; j++) {
            float dot = 0.0f; 
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_tile[local_row][d] * k_tile[j][d];
            }
            dot *= scale; 

            // casual mask: key position > query position means mask out 
            int q_pos = q_start + local_row; 
            int k_pos = kv_start + j; 
            if (k_pos > q_pos || k_pos >= seq_len) {
                dot = -FLT_MAX;
            }
            
            scores[local_row][j] = dot; 
        }

        // online softmax update 
        // find new max for this block 
        float block_max = -FLT_MAX; 
        for (int j = 0; j < BLOCK_N; j++) {
            if (scores[local_row][j] > block_max) {
                block_max = scores[local_row][j]; 
            }
        }

        // new global max
        float new_max = fmaxf(row_max, block_max);

        // rescale previous accumulator: acc *= exp(old_max - new_max)
        float rescale = expf(row_max - new_max); 
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[d] *= rescale; 
        }
        row_sum *= rescale;

        // compute exp(score - new_max) and accumulate
        for (int j = 0; j < BLOCK_N; j++) {
            float p = expf(scores[local_row][j] - new_max);
            row_sum += p;
            for (int d = 0; d < HEAD_DIM; d++) {
                acc[d] += p * v_tile[j][d];
            }
        }

        row_max = new_max;
        __syncthreads();
    }

    // normalize by sum and write output 
    if (q_start + local_row < seq_len) {
        float inv_sum = 1.0f / row_sum; 
        for (int d = 0; d < HEAD_DIM; d++) {
            O[(q_start + local_row) * HEAD_DIM + d] = acc[d] * inv_sum;
        }
    }
}

// launch flash attention for single head
// Q, K, V, O: all [seq_len, HEAD_DIM]
void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len
) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int num_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
    flash_attn_kernel<<<num_blocks, BLOCK_M>>>(Q, K, V, O, seq_len, scale);
    CUDA_CHECK_LAST();
}