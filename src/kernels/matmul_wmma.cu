#include <cuda_fp16.h>
#include <mma.h>
#include "tensor.h"
#include "config.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARPS_M 2
#define WARPS_N 2
#define BLOCK_M (WARPS_M * WMMA_M)
#define BLOCK_N (WARPS_N * WMMA_N)

// convert fp32 buffer to fp16 on GPU (one-time weight conversion)
__global__ void fp32_to_fp16_kernel(const float* in, half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

void convert_fp32_to_fp16(const float* d_fp32, half* d_fp16, size_t n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fp32_to_fp16_kernel<<<grid, block>>>(d_fp32, d_fp16, n);
    CUDA_CHECK_LAST();
}

// C[M,N] = A[M,K] @ B^T[N,K] using tensor cores
// A is fp32 (activations, converted on the fly — small)
// B is fp16 (weights, pre-converted — no overhead)
// accumulation in fp32
__global__ void matmul_wmma_kernel(
    const float* __restrict__ A,
    const half* __restrict__ B_half,
    float* __restrict__ C,
    int M, int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    int tile_row = blockIdx.y * BLOCK_M + warp_row * WMMA_M;
    int tile_col = blockIdx.x * BLOCK_N + warp_col * WMMA_N;

    int warp_active = (tile_row < M && tile_col < N);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // only A needs conversion (small: BLOCK_M rows)
    // B is already fp16
    __shared__ half a_smem[BLOCK_M][WMMA_K];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int kk = 0; kk < K; kk += WMMA_K) {
        // convert A tile fp32->fp16 (activations, small)
        for (int i = tid; i < BLOCK_M * WMMA_K; i += num_threads) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = blockIdx.y * BLOCK_M + r;
            int gc = kk + c;
            float val = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            a_smem[r][c] = __float2half(val);
        }
        __syncthreads();

        // load A from shared, B directly from global (already fp16)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, &a_smem[warp_row * WMMA_M][0], WMMA_K);

        // B_half is [N,K] row-major, load as col_major for transpose
        int b_row = blockIdx.x * BLOCK_N + warp_col * WMMA_N;
        wmma::load_matrix_sync(b_frag, &B_half[b_row * K + kk], K);

        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    // store to shared, then to global
    __shared__ float c_smem[BLOCK_M][BLOCK_N];
    wmma::store_matrix_sync(&c_smem[warp_row * WMMA_M][warp_col * WMMA_N],
                            acc, BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    if (warp_active) {
        int lane = tid % 32;
        for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
            int r = i / WMMA_N;
            int c = i % WMMA_N;
            int gr = tile_row + r;
            int gc = tile_col + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = c_smem[warp_row * WMMA_M + r][warp_col * WMMA_N + c];
            }
        }
    }
}

// public API: takes fp32 A and pre-converted fp16 B
void matmul_wmma_forward(const float* A, const half* B_half, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));
    dim3 block(WARPS_M * WARPS_N * 32);
    matmul_wmma_kernel<<<grid, block>>>(A, B_half, C, M, N, K);
    CUDA_CHECK_LAST();
}
