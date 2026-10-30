#include "tensor.h"
#include "config.h"

#define TILE 32

// C[M,N] = A[M,K] @ B^T[N,K]
// weights stored as [out_features, in_features], so we do B^T access
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
int M, int N, int K) {
    __shared__ float As[TILE][TILE]; 
    __shared__ float Bs[TILE][TILE]; 

    int row = blockIdx.y * TILE + threadIdx.y; 
    int col = blockIdx.x * TILE + threadIdx.x; 

    float sum = 0.0f;

    for (int t = 0 ; t < K ; t += TILE) { 
        // load tile of A
        if (row < M && (t + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        } else { 
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile B^t (B is [N,K] accessing row = col)
        if (col < N && (t + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t + threadIdx.y]; 
        } else { 
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0 ; i < TILE ; i++ ) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_forward(const float* A, const float* B, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    dim3 block(TILE, TILE);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}

// C[M,N] = A[M,K] @ B[K,N] (no transpose on B)
// used for scores @ V where V is an activation, not a weight
__global__ void matmul_nn_kernel(const float* A, const float* B, float* C,
int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE) {
        // load tile of A[M,K]
        if (row < M && (t + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load tile of B[K,N] (no transpose)
        if (col < N && (t + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < TILE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_nn_forward(const float* A, const float* B, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    dim3 block(TILE, TILE);
    matmul_nn_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}

// C[M,N] = A^T[M,K] @ B[K,N]  where A is stored as [K,M]
// used for gradient accumulation: dW = dout^T @ x
__global__ void matmul_tn_kernel(const float* A, const float* B, float* C,
int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;  // M dim
    int col = blockIdx.x * TILE + threadIdx.x;  // N dim

    float sum = 0.0f;
    for (int t = 0; t < K; t += TILE) {
        // A is [K,M]: A[k, row] = A[k*M + row]
        if (row < M && (t + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[(t + threadIdx.x) * M + row];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // B is [K,N]: B[k, col] = B[k*N + col]
        if (col < N && (t + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

void matmul_tn_forward(const float* A, const float* B, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, TILE), CEIL_DIV(M, TILE));
    dim3 block(TILE, TILE);
    matmul_tn_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}