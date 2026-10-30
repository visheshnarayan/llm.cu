#include "tensor.h"
#include "config.h"
#include <math.h>
#include <float.h>

// Cross-entropy loss over all non-padding positions.
// logits: [seq, VOCAB_SIZE]  (last position predicts next token, so we use positions 0..seq-2)
// targets: [seq]             (token ids; target[i] = token at position i+1)
// dlogits: [seq, VOCAB_SIZE] (gradient output; same shape as logits)
// returns: mean cross-entropy loss over seq-1 valid positions

// one block per position: computes softmax + NLL and writes gradient
__global__ void cross_entropy_kernel(const float* logits, const int* targets,
                                     float* dlogits, float* loss_per_pos,
                                     int seq_len, int vocab) {
    // each block handles one query position i predicting token targets[i]
    int i = blockIdx.x;
    if (i >= seq_len - 1) {   // last position has no next token to predict
        if (threadIdx.x == 0) loss_per_pos[i] = 0.0f;
        return;
    }

    const float* row = logits + i * vocab;
    float*       drow = dlogits + i * vocab;
    int target = targets[i];   // predict token at position i+1 from position i's logits

    // -- pass 1: find max for numerical stability --
    __shared__ float smax;
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x)
        local_max = fmaxf(local_max, row[j]);

    __shared__ float red[256];
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) smax = red[0];
    __syncthreads();

    // -- pass 2: sum of exp(x - max) --
    __shared__ float ssum;
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < vocab; j += blockDim.x)
        local_sum += expf(row[j] - smax);

    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) ssum = red[0];
    __syncthreads();

    float log_sum = logf(ssum);

    // -- loss: -log p(target) = -(logit[target] - max - log_sum) --
    if (threadIdx.x == 0)
        loss_per_pos[i] = -(row[target] - smax - log_sum);

    // -- gradient: dL/d(logit[j]) = softmax[j] - 1{j==target}  (for this position) --
    // divide by (seq_len - 1) to get mean over positions
    float norm = 1.0f / (seq_len - 1);
    for (int j = threadIdx.x; j < vocab; j += blockDim.x) {
        float p = expf(row[j] - smax) / ssum;
        drow[j] = (p - (j == target ? 1.0f : 0.0f)) * norm;
    }
}

// host wrapper: returns mean loss, writes dlogits
float cross_entropy_forward(const float* logits, const int* targets,
                             float* dlogits, int seq_len) {
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, seq_len * sizeof(float)));

    cross_entropy_kernel<<<seq_len, 256>>>(logits, targets, dlogits, d_loss, seq_len, VOCAB_SIZE);
    CUDA_CHECK_LAST();

    // sum losses on host
    float* h_loss = (float*)malloc(seq_len * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < seq_len - 1; i++) total += h_loss[i];
    free(h_loss);
    cudaFree(d_loss);

    return total / (seq_len - 1);
}
