#include "tensor.h"
#include "config.h"
#include <math.h>

// per-channel INT8 quantization: find scale per row, quantize to int8
// weight: [rows, cols] fp32 -> q_weight: [rows, cols] int8, scales: [rows] fp32
// scale = max(abs(row)) / 127
__global__ void quantize_kernel(
    const float* weight, int8_t* q_weight, float* scales,
    int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // find max abs value in this row
    __shared__ float smax[256];
    float local_max = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = fabsf(weight[row * cols + c]);
        if (val > local_max) local_max = val;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smax[tid + s] > smax[tid]) {
            smax[tid] = smax[tid + s];
        }
        __syncthreads();
    }

    float scale = smax[0] / 127.0f;
    if (tid == 0) scales[row] = scale;
    __syncthreads();

    // quantize
    float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float val = weight[row * cols + c] * inv_scale;
        // clamp to [-127, 127]
        val = fminf(fmaxf(val, -127.0f), 127.0f);
        q_weight[row * cols + c] = (int8_t)rintf(val);
    }
}

void quantize_weights(const float* weight, int8_t* q_weight, float* scales,
                      int rows, int cols) {
    quantize_kernel<<<rows, 256>>>(weight, q_weight, scales, rows, cols);
    CUDA_CHECK_LAST();
}

// INT8 matmul: C[M,N] = A[M,K](fp32) @ B^T[N,K](int8) * scales[N]
// A is fp32 activation, B is quantized weight with per-row scales
// output C is fp32
__global__ void matmul_int8_kernel(
    const float* A, const int8_t* B_q, const float* B_scales, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        float scale = B_scales[col];
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * (float)B_q[col * K + k];
        }
        C[row * N + col] = sum * scale;
    }
}

void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales,
                          float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(N, 16), CEIL_DIV(M, 16));
    matmul_int8_kernel<<<grid, block>>>(A, B_q, B_scales, C, M, N, K);
    CUDA_CHECK_LAST();
}
