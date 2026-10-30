#include "tensor.h"
#include "config.h"
#include "lora.h"
#include <string.h>

// forward declarations
void embed_forward(const int* token_ids, const float* embed_table, float* out, int seq_len);
void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols);
void rope_forward(float* q, float* k, int seq_len);
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_nn_forward(const float* A, const float* B, float* C, int M, int N, int K);
void residual_forward(const float* x, const float* residual, float* out, int n);
void silu_forward(const float* gate, const float* up, float* out, int n);
void lora_forward_add(const float* x, const LoraAdapter* a,
                      float* y, float* h_buf, int seq);

// per-layer weights struct (mirrors main.cu)
typedef struct {
    float* attn_norm;
    float* ffn_norm;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* q_norm;
    float* k_norm;
    float* w_gate;
    float* w_up;
    float* w_down;
    // fp16 and int8 fields are present but unused in training path
    void* _pad[21];
} LW;

typedef struct {
    float* embed_table;
    LW layers[NUM_LAYERS];
    float* final_norm;
    float* lm_head;
    void* _pad[3];
} MW;

// ---- naive attention that also stores softmax weights ----------------------
// Q: [seq, NUM_HEADS*HEAD_DIM], K: [seq, NUM_HEADS*HEAD_DIM] (expanded),
// V: [seq, NUM_HEADS*HEAD_DIM] (expanded)
// O: [seq, NUM_HEADS*HEAD_DIM]
// attn_w: [NUM_HEADS, seq, seq]  stored softmax weights

__global__ void naive_attn_kernel(const float* Q, const float* K, const float* V,
                                   float* O, float* attn_w,
                                   int seq_len) {
    // one block per (head, query_pos) pair
    int h   = blockIdx.y;
    int qi  = blockIdx.x;
    int tid = threadIdx.x;

    // shared mem: scores[seq_len] and partial sums for softmax
    extern __shared__ float smem[];
    float* scores = smem;  // [seq_len]

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    // compute dot products for all key positions j <= qi (causal)
    float local_max = -1e30f;
    for (int j = tid; j <= qi; j += blockDim.x) {
        int q_base = qi * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
        int k_base = j  * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            dot += Q[q_base + d] * K[k_base + d];
        scores[j] = dot * scale;
        local_max  = fmaxf(local_max, scores[j]);
    }
    // mask future positions
    for (int j = qi + 1 + tid; j < seq_len; j += blockDim.x)
        scores[j] = -1e30f;
    __syncthreads();

    // parallel max reduction
    __shared__ float sred[256];
    sred[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] = fmaxf(sred[tid], sred[tid + s]);
        __syncthreads();
    }
    float gmax = sred[0];

    // exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < seq_len; j += blockDim.x) {
        scores[j] = expf(scores[j] - gmax);
        local_sum += scores[j];
    }
    sred[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sred[tid] += sred[tid + s];
        __syncthreads();
    }
    float gsum = sred[0];

    // normalize and store softmax weights
    for (int j = tid; j < seq_len; j += blockDim.x) {
        float w = scores[j] / gsum;
        scores[j] = w;
        attn_w[h * seq_len * seq_len + qi * seq_len + j] = w;
    }
    __syncthreads();

    // compute output: O[qi, h, d] = sum_j scores[j] * V[j, h, d]
    int o_base = qi * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int v_base = j * NUM_HEADS * HEAD_DIM + h * HEAD_DIM;
            acc += scores[j] * V[v_base + d];
        }
        O[o_base + d] = acc;
    }
}

static void naive_attention_forward_store(
    const float* Q,       // [seq, NUM_HEADS*HEAD_DIM]
    const float* K_exp,   // [seq, NUM_HEADS*HEAD_DIM] (GQA-expanded)
    const float* V_exp,   // [seq, NUM_HEADS*HEAD_DIM]
    float* O,             // [seq, NUM_HEADS*HEAD_DIM]
    float* attn_w,        // [NUM_HEADS, seq, seq]
    int seq_len
) {
    dim3 grid(seq_len, NUM_HEADS);
    dim3 block(256);
    size_t smem = seq_len * sizeof(float);
    naive_attn_kernel<<<grid, block, smem>>>(Q, K_exp, V_exp, O, attn_w, seq_len);
    CUDA_CHECK_LAST();
}

// GQA expand: [seq, NUM_KV_HEADS, HEAD_DIM] -> [seq, NUM_HEADS, HEAD_DIM]
__global__ void expand_kv_lora(const float* kv_in, float* kv_out, int seq_len) {
    int pos  = blockIdx.x;
    int head = blockIdx.y;
    int d    = threadIdx.x;
    int kv_head = head / KV_HEAD_GROUPS;
    if (d < HEAD_DIM) {
        kv_out[pos * NUM_HEADS * HEAD_DIM + head * HEAD_DIM + d] =
            kv_in [pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d];
    }
}

// GQA contract: sum grouped Q gradients back to KV head space
// [seq, NUM_HEADS, HEAD_DIM] -> [seq, NUM_KV_HEADS, HEAD_DIM]
__global__ void contract_kv_grad(const float* grad_exp, float* grad_kv, int seq_len) {
    int pos    = blockIdx.x;
    int kv_h   = blockIdx.y;
    int d      = threadIdx.x;
    if (d < HEAD_DIM) {
        float acc = 0.0f;
        for (int g = 0; g < KV_HEAD_GROUPS; g++)
            acc += grad_exp[pos * NUM_HEADS * HEAD_DIM + (kv_h * KV_HEAD_GROUPS + g) * HEAD_DIM + d];
        grad_kv[pos * NUM_KV_HEADS * HEAD_DIM + kv_h * HEAD_DIM + d] = acc;
    }
}

// ---- transformer_forward_lora ----------------------------------------------
// Same as transformer_forward but:
//   * Uses FP32 matmuls (not WMMA) for training precision
//   * Adds LoRA adapters to Q/K/V/O projections
//   * Stores all activations needed for backward into `acts`

void transformer_forward_lora(
    const int*          token_ids,
    const MW*           model,
    const LoraWeights*  lora,
    float*              hidden,    // [seq, H]   (pre-allocated workspace)
    float*              residual,  // [seq, H]
    float*              norm_out,  // [seq, H]
    float*              q_ws,      // [seq, NUM_HEADS*HEAD_DIM]
    float*              k_ws,      // [seq, NUM_KV_HEADS*HEAD_DIM]
    float*              v_ws,      // [seq, NUM_KV_HEADS*HEAD_DIM]
    float*              k_exp_ws,  // [seq, NUM_HEADS*HEAD_DIM]
    float*              v_exp_ws,  // [seq, NUM_HEADS*HEAD_DIM]
    float*              attn_out,  // [seq, NUM_HEADS*HEAD_DIM]
    float*              gate_buf,  // [seq, INTERMEDIATE_SIZE]
    float*              up_buf,    // [seq, INTERMEDIATE_SIZE]
    float*              logits,    // [seq, VOCAB_SIZE]
    LoraActs*           acts,
    int                 seq_len
) {
    size_t S  = seq_len;
    int Q_DIM = NUM_HEADS    * HEAD_DIM;
    int K_DIM = NUM_KV_HEADS * HEAD_DIM;
    int H     = HIDDEN_SIZE;
    int I     = INTERMEDIATE_SIZE;

    // 1) embedding
    embed_forward(token_ids, model->embed_table, hidden, seq_len);

    // 2) transformer layers
    for (int l = 0; l < NUM_LAYERS; l++) {
        const LW* lw = &model->layers[l];

        // store layer input
        CUDA_CHECK(cudaMemcpy(acts->layer_in[l], hidden,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // save residual
        CUDA_CHECK(cudaMemcpy(residual, hidden,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // attention norm
        rmsnorm_forward(hidden, lw->attn_norm, norm_out, seq_len, H);
        CUDA_CHECK(cudaMemcpy(acts->attn_norm_out[l], norm_out,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // Q/K/V projections (FP32) + LoRA
        matmul_forward(norm_out, lw->wq, q_ws, seq_len, Q_DIM, H);
        lora_forward_add(norm_out, &lora->wq[l], q_ws, acts->hq[l], seq_len);

        matmul_forward(norm_out, lw->wk, k_ws, seq_len, K_DIM, H);
        lora_forward_add(norm_out, &lora->wk[l], k_ws, acts->hk[l], seq_len);

        matmul_forward(norm_out, lw->wv, v_ws, seq_len, K_DIM, H);
        lora_forward_add(norm_out, &lora->wv[l], v_ws, acts->hv[l], seq_len);

        // store QKV before q_norm/k_norm
        CUDA_CHECK(cudaMemcpy(acts->q_raw[l], q_ws,
                              S * Q_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(acts->k_raw[l], k_ws,
                              S * K_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(acts->v_raw[l], v_ws,
                              S * K_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

        // per-head qk-norm (Qwen3 uses RMSNorm on each head)
        rmsnorm_forward(q_ws, lw->q_norm, q_ws, seq_len * NUM_HEADS,    HEAD_DIM);
        rmsnorm_forward(k_ws, lw->k_norm, k_ws, seq_len * NUM_KV_HEADS, HEAD_DIM);

        // RoPE
        rope_forward(q_ws, k_ws, seq_len);

        // store post-norm+rope Q and K
        CUDA_CHECK(cudaMemcpy(acts->q_rope[l], q_ws,
                              S * Q_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(acts->k_rope[l], k_ws,
                              S * K_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

        // GQA expand K, V
        dim3 exp_grid(seq_len, NUM_HEADS);
        expand_kv_lora<<<exp_grid, HEAD_DIM>>>(k_ws, k_exp_ws, seq_len);
        CUDA_CHECK_LAST();
        expand_kv_lora<<<exp_grid, HEAD_DIM>>>(v_ws, v_exp_ws, seq_len);
        CUDA_CHECK_LAST();

        // naive attention (stores softmax weights)
        naive_attention_forward_store(q_ws, k_exp_ws, v_exp_ws,
                                      attn_out, acts->attn_w[l], seq_len);

        // store pre-O-proj attention output
        CUDA_CHECK(cudaMemcpy(acts->pre_o[l], attn_out,
                              S * Q_DIM * sizeof(float), cudaMemcpyDeviceToDevice));

        // O projection + LoRA
        matmul_forward(attn_out, lw->wo, norm_out, seq_len, H, Q_DIM);
        lora_forward_add(attn_out, &lora->wo[l], norm_out, acts->ho[l], seq_len);

        // attention residual
        residual_forward(norm_out, residual, hidden, seq_len * H);

        // store post-attention hidden
        CUDA_CHECK(cudaMemcpy(acts->post_attn[l], hidden,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // FFN residual save
        CUDA_CHECK(cudaMemcpy(residual, hidden,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // FFN norm
        rmsnorm_forward(hidden, lw->ffn_norm, norm_out, seq_len, H);
        CUDA_CHECK(cudaMemcpy(acts->ffn_norm_out[l], norm_out,
                              S * H * sizeof(float), cudaMemcpyDeviceToDevice));

        // gate + up projections (FP32, no LoRA on MLP)
        matmul_forward(norm_out, lw->w_gate, gate_buf, seq_len, I, H);
        matmul_forward(norm_out, lw->w_up,   up_buf,   seq_len, I, H);

        // store pre-activation values for backward
        CUDA_CHECK(cudaMemcpy(acts->gate_raw[l], gate_buf,
                              S * I * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(acts->up_raw[l], up_buf,
                              S * I * sizeof(float), cudaMemcpyDeviceToDevice));

        // SwiGLU: gate_buf <- silu(gate) * up
        silu_forward(gate_buf, up_buf, gate_buf, seq_len * I);

        // down projection
        matmul_forward(gate_buf, lw->w_down, attn_out, seq_len, H, I);

        // MLP residual
        residual_forward(attn_out, residual, hidden, seq_len * H);
    }

    // 3) final norm
    rmsnorm_forward(hidden, model->final_norm, hidden, seq_len, H);

    // 4) logits: hidden [seq, H] @ lm_head^T [VOCAB, H]
    matmul_forward(hidden, model->lm_head, logits, seq_len, VOCAB_SIZE, H);
}
