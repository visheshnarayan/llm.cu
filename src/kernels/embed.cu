#include "tensor.h"
#include "config.h"

// grid=(seq_len, CEIL_DIV(HIDDEN_SIZE, 256)), block=256
// each thread copies one element from embedding table
__global__ void embed_kernel(const int* token_ids, const float* embed_table, float* out, int seq_len) {
    int pos = blockIdx.x; // which tok 
    int col = blockIdx.y * blockDim.x + threadIdx.x; // which hidden dim 

    if (pos < seq_len && col < HIDDEN_SIZE) {
        int token_id = token_ids[pos]; 
        out[pos * HIDDEN_SIZE + col] = embed_table[token_id * HIDDEN_SIZE + col]; 
    }
}

void embed_forward(const int* token_ids, const float* embed_table, float* out, int seq_len) { 
    dim3 grid(seq_len, CEIL_DIV(HIDDEN_SIZE, 256));
    dim3 block(256); 
    embed_kernel<<<grid, block>>>(token_ids, embed_table, out, seq_len); 
    CUDA_CHECK_LAST(); 
}