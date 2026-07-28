#include "tensor.h"
#include "config.h"
#include <mma.h>

using namespace nvcuda;

// wmma tile dimensions (fixed by hardware)
// each warp computes a 16x16 output tile using 16x16 @ 16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// block tile: each block has 8 warps (256 threads)
// block computes a 64x64 output tile (4x4 warp tiles)
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define WARPS_M 4
#define WARPS_N 4
#define NUM_WARPS (WARPS_M * WARPS_N)

// C[M,N] = A[M,K] @ B^T[N,K] using tensor cores
// A is fp32 input, converted to fp16 on the fly
// accumulation in fp32 for precision
__global__ void matmul_wmma_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // which warp in the block
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / WARPS_N;  // 0..3
    int warp_col = warp_id % WARPS_N;  // 0..3

    // global tile position
    int block_row = blockIdx.y * BLOCK_TILE_M;
    int block_col = blockIdx.x * BLOCK_TILE_N;

    // this warp's output position
    int warp_m = block_row + warp_row * WMMA_M;
    int warp_n = block_col + warp_col * WMMA_N;

    // accumulator fragment (fp32)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // loop over K in WMMA_K chunks
    for (int kk = 0; kk < K; kk += WMMA_K) {
        // declare fp16 input fragments
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        // load A tile [warp_m:warp_m+16, kk:kk+16] as row_major
        // need to convert fp32 -> fp16 in shared memory first
        __shared__ half a_shared[BLOCK_TILE_M][WMMA_K];
        __shared__ half b_shared[BLOCK_TILE_N][WMMA_K];

        // cooperative load: all threads in block help convert fp32->fp16
        int tid = threadIdx.x;
        int total_threads = blockDim.x;

        // load A block tile rows into shared
        int a_elems = BLOCK_TILE_M * WMMA_K;
        for (int i = tid; i < a_elems; i += total_threads) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int global_row = block_row + row;
            int global_col = kk + col;
            if (global_row < M && global_col < K) {
                a_shared[row][col] = __float2half(A[global_row * K + global_col]);
            } else {
                a_shared[row][col] = __float2half(0.0f);
            }
        }

        // load B block tile: B is [N,K], we load rows for this block's columns
        int b_elems = BLOCK_TILE_N * WMMA_K;
        for (int i = tid; i < b_elems; i += total_threads) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int global_row = block_col + row;
            int global_col = kk + col;
            if (global_row < N && global_col < K) {
                b_shared[row][col] = __float2half(B[global_row * K + global_col]);
            } else {
                b_shared[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // load fragments from shared memory
        if (warp_m < M && warp_n < N) {
            wmma::load_matrix_sync(a_frag, &a_shared[warp_row * WMMA_M][0], WMMA_K);
            // B is col_major: B^T[K,N] stored as B[N,K], load as col_major
            wmma::load_matrix_sync(b_frag, &b_shared[warp_col * WMMA_N][0], WMMA_K);

            // tensor core multiply-accumulate
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        __syncthreads();
    }

    // store accumulator to global memory
    if (warp_m < M && warp_n < N) {
        // store to shared first, then to global (wmma requires aligned store)
        __shared__ float c_shared[BLOCK_TILE_M][BLOCK_TILE_N];
        wmma::store_matrix_sync(&c_shared[warp_row * WMMA_M][warp_col * WMMA_N],
                                acc, BLOCK_TILE_N, wmma::mem_row_major);
        __syncthreads();

        // copy from shared to global
        int tid = threadIdx.x;
        int total_threads = blockDim.x;
        for (int i = tid; i < WMMA_M * WMMA_N; i += total_threads) {
            int r = i / WMMA_N;
            int c = i % WMMA_N;
            int gr = warp_m + r;
            int gc = warp_n + c;
            if (gr < M && gc < N) {
                C[gr * N + gc] = c_shared[warp_row * WMMA_M + r][warp_col * WMMA_N + c];
            }
        }
    }
}

void matmul_wmma_forward(const float* A, const float* B, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, BLOCK_TILE_N), CEIL_DIV(M, BLOCK_TILE_M));
    dim3 block(NUM_WARPS * 32);  // 16 warps * 32 = 512 threads
    matmul_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}
