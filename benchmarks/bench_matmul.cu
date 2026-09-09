#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include "tensor.h"
#include "config.h"

// matmul variants
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_wmma_forward(const float* A, const half* B_half, float* C, int M, int N, int K);
void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales, float* C, int M, int N, int K);
void convert_fp32_to_fp16(const float* d_fp32, half* d_fp16, size_t n);
void quantize_weights(const float* weight, int8_t* q_weight, float* scales, int rows, int cols);

// fill buffer with random values
__global__ void fill_random(float* buf, int n, int seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // simple LCG random
        unsigned int x = (unsigned int)(i + seed) * 1103515245u + 12345u;
        buf[i] = ((float)(x & 0xFFFF) / 65536.0f - 0.5f) * 2.0f;
    }
}

float benchmark_kernel(int warmup, int iters, void (*run)(void*), void* args) {
    for (int i = 0; i < warmup; i++) run(args);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) run(args);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

// benchmark args
typedef struct {
    float *A, *B, *C;
    half *B_h;
    int8_t *B_q;
    float *B_scales;
    int M, N, K;
} BenchArgs;

void run_fp32(void* p) {
    BenchArgs* a = (BenchArgs*)p;
    matmul_forward(a->A, a->B, a->C, a->M, a->N, a->K);
}

void run_wmma(void* p) {
    BenchArgs* a = (BenchArgs*)p;
    matmul_wmma_forward(a->A, a->B_h, a->C, a->M, a->N, a->K);
}

void run_int8(void* p) {
    BenchArgs* a = (BenchArgs*)p;
    matmul_int8_forward(a->A, a->B_q, a->B_scales, a->C, a->M, a->N, a->K);
}

void bench_size(int M, int N, int K, FILE* csv) {
    BenchArgs args;
    args.M = M; args.N = N; args.K = K;

    // allocate
    CUDA_CHECK(cudaMalloc(&args.A, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&args.B, (size_t)N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&args.C, (size_t)M * N * sizeof(float)));

    // fill with random data
    int grid = CEIL_DIV(M * K, 256);
    fill_random<<<grid, 256>>>(args.A, M * K, 42);
    grid = CEIL_DIV(N * K, 256);
    fill_random<<<grid, 256>>>(args.B, N * K, 123);
    CUDA_CHECK(cudaDeviceSynchronize());

    // prep fp16 weights
    CUDA_CHECK(cudaMalloc(&args.B_h, (size_t)N * K * sizeof(half)));
    convert_fp32_to_fp16(args.B, args.B_h, (size_t)N * K);

    // prep int8 weights
    CUDA_CHECK(cudaMalloc(&args.B_q, (size_t)N * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&args.B_scales, (size_t)N * sizeof(float)));
    quantize_weights(args.B, args.B_q, args.B_scales, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    double flops = 2.0 * M * N * K;
    int warmup = 5, iters = 20;

    // benchmark each variant
    float ms_fp32 = benchmark_kernel(warmup, iters, run_fp32, &args);
    float ms_wmma = benchmark_kernel(warmup, iters, run_wmma, &args);
    float ms_int8 = benchmark_kernel(warmup, iters, run_int8, &args);

    double gflops_fp32 = flops / (ms_fp32 * 1e6);
    double gflops_wmma = flops / (ms_wmma * 1e6);
    double gflops_int8 = flops / (ms_int8 * 1e6);

    printf("  [%4dx%4dx%4d]  FP32: %7.3f ms (%7.1f GFLOPS)  "
           "WMMA: %7.3f ms (%7.1f GFLOPS)  "
           "INT8: %7.3f ms (%7.1f GFLOPS)\n",
           M, N, K,
           ms_fp32, gflops_fp32,
           ms_wmma, gflops_wmma,
           ms_int8, gflops_int8);

    fprintf(csv, "%d,%d,%d,%.4f,%.1f,%.4f,%.1f,%.4f,%.1f\n",
            M, N, K,
            ms_fp32, gflops_fp32,
            ms_wmma, gflops_wmma,
            ms_int8, gflops_int8);

    cudaFree(args.A); cudaFree(args.B); cudaFree(args.C);
    cudaFree(args.B_h); cudaFree(args.B_q); cudaFree(args.B_scales);
}

int main() {
    printf("Matmul Benchmark: FP32 vs WMMA(FP16) vs INT8\n");
    printf("=============================================\n\n");

    FILE* csv = fopen("data/bench_matmul.csv", "w");
    fprintf(csv, "M,N,K,fp32_ms,fp32_gflops,wmma_ms,wmma_gflops,int8_ms,int8_gflops\n");

    // qwen3 matmul shapes
    printf("Qwen3 projection shapes:\n");
    bench_size(1, 2048, 2048, csv);    // decode: Q projection
    bench_size(1, 6144, 2048, csv);    // decode: gate/up projection
    bench_size(1, 2048, 6144, csv);    // decode: down projection
    bench_size(1, 151936, 2048, csv);  // decode: lm_head

    printf("\nPrefill shapes (seq=128):\n");
    bench_size(128, 2048, 2048, csv);    // prefill: Q projection
    bench_size(128, 6144, 2048, csv);    // prefill: gate/up
    bench_size(128, 2048, 6144, csv);    // prefill: down
    bench_size(128, 151936, 2048, csv);  // prefill: lm_head

    printf("\nLarger batch (seq=512):\n");
    bench_size(512, 2048, 2048, csv);
    bench_size(512, 6144, 2048, csv);
    bench_size(512, 2048, 6144, csv);

    fclose(csv);
    printf("\nResults saved to data/bench_matmul.csv\n");
    return 0;
}
