#ifndef LORA_H
#define LORA_H

#include "config.h"

#define LORA_RANK   16
#define LORA_ALPHA  32
#define LORA_SCALE  2.0f   // LORA_ALPHA / LORA_RANK

// ---- structs ---------------------------------------------------------------

// one LoRA adapter: A[r, d_in], B[d_out, r]
typedef struct {
    float* A;           // [LORA_RANK, d_in]
    float* B;           // [d_out, LORA_RANK]
    float* mA; float* vA;   // AdamW first/second moments
    float* mB; float* vB;
    int d_in, d_out;
} LoraAdapter;

// adapters on Q/K/V/O projections for every layer
typedef struct {
    LoraAdapter wq[NUM_LAYERS];   // d_out=NUM_HEADS*HEAD_DIM,   d_in=HIDDEN_SIZE
    LoraAdapter wk[NUM_LAYERS];   // d_out=NUM_KV_HEADS*HEAD_DIM, d_in=HIDDEN_SIZE
    LoraAdapter wv[NUM_LAYERS];   // d_out=NUM_KV_HEADS*HEAD_DIM, d_in=HIDDEN_SIZE
    LoraAdapter wo[NUM_LAYERS];   // d_out=HIDDEN_SIZE,           d_in=NUM_HEADS*HEAD_DIM
} LoraWeights;

// gradient accumulators (same shape as adapters, no moment buffers)
typedef struct {
    float* dA;  // [LORA_RANK, d_in]
    float* dB;  // [d_out, LORA_RANK]
} LoraGrad;

typedef struct {
    LoraGrad wq[NUM_LAYERS];
    LoraGrad wk[NUM_LAYERS];
    LoraGrad wv[NUM_LAYERS];
    LoraGrad wo[NUM_LAYERS];
} LoraGrads;

// activations stored during forward pass (needed for backward)
typedef struct {
    float* layer_in[NUM_LAYERS];       // [seq, HIDDEN_SIZE]           input to layer l
    float* attn_norm_out[NUM_LAYERS];  // [seq, HIDDEN_SIZE]           after attn rmsnorm
    float* q_raw[NUM_LAYERS];          // [seq, NUM_HEADS*HEAD_DIM]    before q_norm
    float* k_raw[NUM_LAYERS];          // [seq, NUM_KV_HEADS*HEAD_DIM] before k_norm
    float* q_rope[NUM_LAYERS];         // [seq, NUM_HEADS*HEAD_DIM]    after q_norm+rope
    float* k_rope[NUM_LAYERS];         // [seq, NUM_KV_HEADS*HEAD_DIM] after k_norm+rope
    float* v_raw[NUM_LAYERS];          // [seq, NUM_KV_HEADS*HEAD_DIM]
    float* attn_w[NUM_LAYERS];         // [NUM_HEADS, seq, seq]        softmax weights
    float* pre_o[NUM_LAYERS];          // [seq, NUM_HEADS*HEAD_DIM]    before o_proj
    float* post_attn[NUM_LAYERS];      // [seq, HIDDEN_SIZE]           after attn residual
    float* ffn_norm_out[NUM_LAYERS];   // [seq, HIDDEN_SIZE]           after ffn rmsnorm
    float* gate_raw[NUM_LAYERS];       // [seq, INTERMEDIATE_SIZE]     before silu
    float* up_raw[NUM_LAYERS];         // [seq, INTERMEDIATE_SIZE]
    // LoRA h = A*x per projection
    float* hq[NUM_LAYERS];  // [seq, LORA_RANK]
    float* hk[NUM_LAYERS];
    float* hv[NUM_LAYERS];
    float* ho[NUM_LAYERS];
} LoraActs;

// ---- function declarations -------------------------------------------------

void lora_weights_init(LoraWeights* lora);
void lora_weights_free(LoraWeights* lora);

void lora_grads_alloc(LoraGrads* grads, const LoraWeights* lora);
void lora_grads_free(LoraGrads* grads);
void lora_grads_zero(LoraGrads* grads);

void lora_acts_alloc(LoraActs* acts, int seq_len);
void lora_acts_free(LoraActs* acts, int seq_len);

// y[seq, d_out] += LORA_SCALE * B * (A * x)
// stores h = A*x into h_buf[seq, LORA_RANK] if h_buf != NULL
void lora_forward_add(const float* x, const LoraAdapter* a,
                      float* y, float* h_buf, int seq);

// AdamW update for all LoRA parameters
void adamw_step_lora(LoraWeights* lora, const LoraGrads* grads,
                     float lr, float beta1, float beta2, float eps,
                     float wd, int step);

// element-wise: dst += scale * src  (n elements)
void scale_add_inplace(float* dst, const float* src, float scale, int n);

// zeros a float buffer on device
void gpu_zero(float* buf, int n);

#endif // LORA_H
