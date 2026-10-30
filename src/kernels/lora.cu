#include "tensor.h"
#include "config.h"
#include "lora.h"
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

// ---- tiny helpers ----------------------------------------------------------

__global__ void scale_add_kernel(float* dst, const float* src, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += scale * src[i];
}

void scale_add_inplace(float* dst, const float* src, float scale, int n) {
    scale_add_kernel<<<CEIL_DIV(n, 256), 256>>>(dst, src, scale, n);
    CUDA_CHECK_LAST();
}

__global__ void zero_kernel(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = 0.0f;
}

void gpu_zero(float* buf, int n) {
    zero_kernel<<<CEIL_DIV(n, 256), 256>>>(buf, n);
    CUDA_CHECK_LAST();
}

// ---- random init: A ~ N(0, 0.02), B = 0 ------------------------------------

__global__ void randn_init_kernel(float* A, int n, float std, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState state;
    curand_init(seed, i, 0, &state);
    A[i] = curand_normal(&state) * std;
}

static void init_adapter(LoraAdapter* a, int d_out, int d_in) {
    a->d_out = d_out;
    a->d_in  = d_in;
    size_t szA = (size_t)LORA_RANK * d_in;
    size_t szB = (size_t)d_out * LORA_RANK;

    CUDA_CHECK(cudaMalloc(&a->A,  szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->B,  szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mA, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->vA, szA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->mB, szB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a->vB, szB * sizeof(float)));

    // A ~ N(0, 0.02), B = 0
    randn_init_kernel<<<CEIL_DIV(szA, 256), 256>>>(a->A, (int)szA, 0.02f,
                                                    (unsigned long long)(d_out * d_in));
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemset(a->B,  0, szB * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->mA, 0, szA * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->vA, 0, szA * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->mB, 0, szB * sizeof(float)));
    CUDA_CHECK(cudaMemset(a->vB, 0, szB * sizeof(float)));
}

static void free_adapter(LoraAdapter* a) {
    cudaFree(a->A);  cudaFree(a->B);
    cudaFree(a->mA); cudaFree(a->vA);
    cudaFree(a->mB); cudaFree(a->vB);
}

void lora_weights_init(LoraWeights* lora) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        init_adapter(&lora->wq[l], NUM_HEADS    * HEAD_DIM, HIDDEN_SIZE);
        init_adapter(&lora->wk[l], NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        init_adapter(&lora->wv[l], NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        init_adapter(&lora->wo[l], HIDDEN_SIZE,              NUM_HEADS * HEAD_DIM);
    }
}

void lora_weights_free(LoraWeights* lora) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        free_adapter(&lora->wq[l]);
        free_adapter(&lora->wk[l]);
        free_adapter(&lora->wv[l]);
        free_adapter(&lora->wo[l]);
    }
}

// ---- gradient buffers -------------------------------------------------------

static void alloc_grad(LoraGrad* g, int d_out, int d_in) {
    CUDA_CHECK(cudaMalloc(&g->dA, (size_t)LORA_RANK * d_in  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g->dB, (size_t)d_out * LORA_RANK * sizeof(float)));
}

void lora_grads_alloc(LoraGrads* grads, const LoraWeights* lora) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        alloc_grad(&grads->wq[l], lora->wq[l].d_out, lora->wq[l].d_in);
        alloc_grad(&grads->wk[l], lora->wk[l].d_out, lora->wk[l].d_in);
        alloc_grad(&grads->wv[l], lora->wv[l].d_out, lora->wv[l].d_in);
        alloc_grad(&grads->wo[l], lora->wo[l].d_out, lora->wo[l].d_in);
    }
}

void lora_grads_free(LoraGrads* grads) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        cudaFree(grads->wq[l].dA); cudaFree(grads->wq[l].dB);
        cudaFree(grads->wk[l].dA); cudaFree(grads->wk[l].dB);
        cudaFree(grads->wv[l].dA); cudaFree(grads->wv[l].dB);
        cudaFree(grads->wo[l].dA); cudaFree(grads->wo[l].dB);
    }
}

void lora_grads_zero(LoraGrads* grads) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        const LoraGrad* gq = &grads->wq[l];
        const LoraGrad* gk = &grads->wk[l];
        const LoraGrad* gv = &grads->wv[l];
        const LoraGrad* go = &grads->wo[l];
        gpu_zero(gq->dA, LORA_RANK * HIDDEN_SIZE);
        gpu_zero(gq->dB, (NUM_HEADS * HEAD_DIM) * LORA_RANK);
        gpu_zero(gk->dA, LORA_RANK * HIDDEN_SIZE);
        gpu_zero(gk->dB, (NUM_KV_HEADS * HEAD_DIM) * LORA_RANK);
        gpu_zero(gv->dA, LORA_RANK * HIDDEN_SIZE);
        gpu_zero(gv->dB, (NUM_KV_HEADS * HEAD_DIM) * LORA_RANK);
        gpu_zero(go->dA, LORA_RANK * (NUM_HEADS * HEAD_DIM));
        gpu_zero(go->dB, HIDDEN_SIZE * LORA_RANK);
    }
}

// ---- activation storage ----------------------------------------------------

void lora_acts_alloc(LoraActs* acts, int seq_len) {
    size_t S = seq_len;
    for (int l = 0; l < NUM_LAYERS; l++) {
        CUDA_CHECK(cudaMalloc(&acts->layer_in[l],      S * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->attn_norm_out[l], S * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->q_raw[l],  S * NUM_HEADS    * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->k_raw[l],  S * NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->q_rope[l], S * NUM_HEADS    * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->k_rope[l], S * NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->v_raw[l],  S * NUM_KV_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->attn_w[l], (size_t)NUM_HEADS * S * S * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->pre_o[l],  S * NUM_HEADS * HEAD_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->post_attn[l],   S * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->ffn_norm_out[l],S * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->gate_raw[l], S * INTERMEDIATE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->up_raw[l],   S * INTERMEDIATE_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->hq[l], S * LORA_RANK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->hk[l], S * LORA_RANK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->hv[l], S * LORA_RANK * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&acts->ho[l], S * LORA_RANK * sizeof(float)));
    }
}

void lora_acts_free(LoraActs* acts, int seq_len) {
    (void)seq_len;
    for (int l = 0; l < NUM_LAYERS; l++) {
        cudaFree(acts->layer_in[l]);
        cudaFree(acts->attn_norm_out[l]);
        cudaFree(acts->q_raw[l]);    cudaFree(acts->k_raw[l]);
        cudaFree(acts->q_rope[l]);   cudaFree(acts->k_rope[l]);
        cudaFree(acts->v_raw[l]);
        cudaFree(acts->attn_w[l]);
        cudaFree(acts->pre_o[l]);
        cudaFree(acts->post_attn[l]);
        cudaFree(acts->ffn_norm_out[l]);
        cudaFree(acts->gate_raw[l]); cudaFree(acts->up_raw[l]);
        cudaFree(acts->hq[l]); cudaFree(acts->hk[l]);
        cudaFree(acts->hv[l]); cudaFree(acts->ho[l]);
    }
}

// ---- LoRA forward add -------------------------------------------------------
// Uses matmul_forward declared in matmul.cu

void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_nn_forward(const float* A, const float* B, float* C, int M, int N, int K);

// y += LORA_SCALE * B * (A * x)
// h_buf (optional): stores intermediate h = A*x [seq, LORA_RANK]
void lora_forward_add(const float* x, const LoraAdapter* a,
                      float* y, float* h_buf, int seq) {
    // h = x @ A^T  → matmul_forward(x, A, h, seq, r, d_in) gives h[seq,r] = x[seq,d_in]@A^T[d_in,r]
    //   since A is stored as [r, d_in], matmul_forward treats it as B[N,K] → C[M,N]=A[M,K]@B^T → ✓
    float* h;
    bool own_h = (h_buf == NULL);
    if (own_h) CUDA_CHECK(cudaMalloc(&h, (size_t)seq * LORA_RANK * sizeof(float)));
    else        h = h_buf;

    matmul_forward(x, a->A, h, seq, LORA_RANK, a->d_in);

    // tmp = h @ B^T  [seq, d_out] = h[seq,r] @ B^T[d_out, r]
    float* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, (size_t)seq * a->d_out * sizeof(float)));
    matmul_forward(h, a->B, tmp, seq, a->d_out, LORA_RANK);

    // y += LORA_SCALE * tmp
    scale_add_inplace(y, tmp, LORA_SCALE, seq * a->d_out);

    cudaFree(tmp);
    if (own_h) cudaFree(h);
}

// ---- AdamW ------------------------------------------------------------------

__global__ void adamw_kernel(float* param, const float* grad,
                              float* m, float* v, int n,
                              float lr, float beta1, float beta2,
                              float eps, float wd, float bc1, float bc2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    param[i] = param[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);
}

static void adamw_adapter(LoraAdapter* a, const LoraGrad* g,
                          float lr, float beta1, float beta2, float eps, float wd,
                          float bc1, float bc2) {
    int szA = LORA_RANK * a->d_in;
    int szB = a->d_out * LORA_RANK;
    adamw_kernel<<<CEIL_DIV(szA, 256), 256>>>(
        a->A, g->dA, a->mA, a->vA, szA, lr, beta1, beta2, eps, wd, bc1, bc2);
    CUDA_CHECK_LAST();
    adamw_kernel<<<CEIL_DIV(szB, 256), 256>>>(
        a->B, g->dB, a->mB, a->vB, szB, lr, beta1, beta2, eps, wd, bc1, bc2);
    CUDA_CHECK_LAST();
}

void adamw_step_lora(LoraWeights* lora, const LoraGrads* grads,
                     float lr, float beta1, float beta2, float eps,
                     float wd, int step) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int l = 0; l < NUM_LAYERS; l++) {
        adamw_adapter(&lora->wq[l], &grads->wq[l], lr, beta1, beta2, eps, wd, bc1, bc2);
        adamw_adapter(&lora->wk[l], &grads->wk[l], lr, beta1, beta2, eps, wd, bc1, bc2);
        adamw_adapter(&lora->wv[l], &grads->wv[l], lr, beta1, beta2, eps, wd, bc1, bc2);
        adamw_adapter(&lora->wo[l], &grads->wo[l], lr, beta1, beta2, eps, wd, bc1, bc2);
    }
}
