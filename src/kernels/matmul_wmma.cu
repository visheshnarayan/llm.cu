#include <cuda_fp16.h>
#include <mma.h>
#include "tensor.h"
#include "config.h"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// block: 2x2 warp tiles = 32x32 output, 4 warps = 128 threads
#define WARPS_M 2
#define WARPS_N 2
#define BLOCK_M (WARPS_M * WMMA_M)
#define BLOCK_N (WARPS_N * WMMA_N)

// C[M,N] = A[M,K] @ B^T[N,K] using tensor cores
// fp32 in, fp16 compute, fp32 accumulate
__global__ void matmul_wmma_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    int warp_id = threadIdx.x / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;

    int tile_row = blockIdx.y * BLOCK_M + warp_row * WMMA_M;
    int tile_col = blockIdx.x * BLOCK_N + warp_col * WMMA_N;

    // all warps participate in loads/syncs, only active ones store results
    int warp_active = (tile_row < M && tile_col < N);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    __shared__ half a_smem[BLOCK_M][WMMA_K];
    __shared__ half b_smem[BLOCK_N][WMMA_K];

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    for (int kk = 0; kk < K; kk += WMMA_K) {
        // cooperative fp32->fp16 load for A [M,K]
        for (int i = tid; i < BLOCK_M * WMMA_K; i += num_threads) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = blockIdx.y * BLOCK_M + r;
            int gc = kk + c;
            float val = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            a_smem[r][c] = __float2half(val);
        }

        // cooperative fp32->fp16 load for B [N,K]
        for (int i = tid; i < BLOCK_N * WMMA_K; i += num_threads) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gr = blockIdx.x * BLOCK_N + r;
            int gc = kk + c;
            float val = (gr < N && gc < K) ? B[gr * K + gc] : 0.0f;
            b_smem[r][c] = __float2half(val);
        }
        __syncthreads();

        // all warps compute (inactive ones produce zeros, discarded later)
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, &a_smem[warp_row * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &b_smem[warp_col * WMMA_N][0], WMMA_K);
        wmma::mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    // store to shared, then to global
    __shared__ float c_smem[BLOCK_M][BLOCK_N];
    wmma::store_matrix_sync(&c_smem[warp_row * WMMA_M][warp_col * WMMA_N],
                            acc, BLOCK_N, wmma::mem_row_major);
    __syncthreads();

    // only active warps write to global
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

void matmul_wmma_forward(const float* A, const float* B, float* C,
int M, int N, int K) {
    dim3 grid(CEIL_DIV(N, BLOCK_N), CEIL_DIV(M, BLOCK_M));
    dim3 block(WARPS_M * WARPS_N * 32);
    matmul_wmma_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
}
