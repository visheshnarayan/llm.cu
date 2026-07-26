#include "tensor.h"
#include "config.h"
#include <stdio.h>

// debug: save gpu buffer to file
void debug_save(const char* path, const float* dev_ptr, size_t elements) {
    float* host = (float*)malloc(elements * sizeof(float));
    CUDA_CHECK(cudaMemcpy(host, dev_ptr, elements * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* f = fopen(path, "wb");
    fwrite(host, sizeof(float), elements, f);
    fclose(f);
    free(host);
}

// kernel forward declarations
void embed_forward(const int* token_ids, const float* embed_table, float* out, int seq_len); 
void rmsnorm_forward(const float* x, const float* weight, float* out, int rows, int cols); 
void attention_forward(
    const float* hidden, const float* wq, const float* wk, const float* wv, const float* wo,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out, float* q, float* k, float* v, float* k_exp, float* v_exp,
    float* scores, float* attn_out,
    float* q_head, float* k_head, float* v_head, float* o_head,
    int seq_len
);
void mlp_forward(
    const float* hidden, const float* w_gate, const float* w_up, const float* w_down, 
    float* out, float* gate_buf, float* up_buf, int seq_len
);
void residual_forward(const float* x, const float* residual, float* out, int n);
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K); 

// per layer weights (pointers into gpu memory)
typedef struct {
    float* attn_norm;       // [HIDDEN_SIZE]
    float* ffn_norm;        // [HIDDEN_SIZE]
    float* wq;              // [NUM_HEADS * HEAD_DIM, HIDDEN_SIZE]
    float* wk;              // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE]
    float* wv;              // [NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE]
    float* wo;              // [HIDDEN_SIZE, NUM_HEADS * HEAD_DIM]
    float* q_norm;          // [HEAD_DIM]
    float* k_norm;          // [HEAD_DIM]
    float* w_gate;          // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    float* w_up;            // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    float* w_down;          // [HIDDEN_SIZE, INTERMEDIATE_SIZE]
} LayerWeights;

// all model weights
typedef struct {
    float* embed_table;                 // [VOCAB_SIZE, HIDDEN_SIZE]
    LayerWeights layers[NUM_LAYERS];
    float* final_norm;                  // [HIDDEN_SIZE]
    float* lm_head;                     // [VOCAB_SIZE, HIDDEN_SIZE]
} ModelWeights;

// workspace buffers (pre-allocated, reused across layers)
typedef struct {
    float* hidden;          // [seq, HIDDEN_SIZE] :: main hidden state
    float* residual;        // [seq, HIDDEN_SIZE] :: residual connection
    float* norm_out;        // [seq, HIDDEN_SIZE] :: norm output
    float* attn_out;        // [seq, NUM_HEADS * HEAD_DIM] :: attention output 
    float* q;               // [seq, NUM_HEADS * HEAD_DIM]
    float* k;               // [seq, NUM_KV_HEADS * HEAD_DIM]
    float* v;               // [seq, NUM_KV_HEADS * HEAD_DIM]
    float* k_exp;           // [seq, NUM_HEADS * HEAD_DIM]
    float* v_exp;           // [seq, NUM_HEADS * HEAD_DIM]
    float* scores;          // [NUM_HEADS, seq, seq]
    float* attn_work;       // [seq, NUM_HEADS * HEAD_DIM]
    float* q_head;          // [seq, HEAD_DIM] :: per-head workspace
    float* k_head;          // [seq, HEAD_DIM]
    float* v_head;          // [seq, HEAD_DIM]
    float* o_head;          // [seq, HEAD_DIM]
    float* gate_buf;        // [seq, INTERMEDIATE_SIZE]
    float* up_buf;          // [seq, INTERMEDIATE_SIZE]
    float* logits;          // [seq, VOCAB_SIZE]
} Workspace; 

void transformer_forward(
    const int* token_ids, 
    const ModelWeights* model, 
    Workspace* ws, 
    int seq_len
) { 
    int hidden_total = seq_len * HIDDEN_SIZE; 

    // 1) embedding lookup
    embed_forward(token_ids, model->embed_table, ws->hidden, seq_len);
    debug_save("data/cuda_out/embed_out.bin", ws->hidden, (size_t)seq_len * HIDDEN_SIZE);

    // 2) transformer layers
    char debug_path[256];
    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l]; 

        // save residual 
        CUDA_CHECK(
            cudaMemcpy(ws->residual, ws->hidden, 
                hidden_total * sizeof(float), cudaMemcpyDeviceToDevice
            )
        );

        // attention norm
        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);
        sprintf(debug_path, "data/cuda_out/layer_%02d_attn_norm.bin", l);
        debug_save(debug_path, ws->norm_out, (size_t)seq_len * HIDDEN_SIZE);

        // self-attn 
        attention_forward(
            ws->norm_out, lw->wq, lw->wk, lw->wv, lw->wo,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->k_exp, ws->v_exp,
            ws->scores, ws->attn_work,
            ws->q_head, ws->k_head, ws->v_head, ws->o_head,
            seq_len
        );
        
        // residual after attention 
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);

        // save residual for MLP 
        CUDA_CHECK(
            cudaMemcpy(ws->residual, ws->hidden,
                hidden_total * sizeof(float), cudaMemcpyDeviceToDevice
            )
        );

        // ffn norm 
        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);

        // MLP (SwiGLU)
        mlp_forward(ws->norm_out, lw->w_gate, lw->w_up, lw->w_down, 
                    ws->attn_out, ws->gate_buf, ws->up_buf, seq_len);
        
        // residual after MLP
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
        sprintf(debug_path, "data/cuda_out/layer_%02d_out.bin", l);
        debug_save(debug_path, ws->hidden, (size_t)seq_len * HIDDEN_SIZE);
    }

    // 3) final RMSNorm
    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, seq_len, HIDDEN_SIZE);
    debug_save("data/cuda_out/final_norm.bin", ws->hidden, (size_t)seq_len * HIDDEN_SIZE);

    // 4) logits: hidden @ lm_head^T
    matmul_forward(ws->hidden, model->lm_head, ws->logits, seq_len, VOCAB_SIZE, HIDDEN_SIZE);
    debug_save("data/cuda_out/logits.bin", ws->logits, (size_t)seq_len * VOCAB_SIZE);
}