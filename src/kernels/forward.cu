#include "tensor.h"
#include "config.h"
#include <stdio.h>
#include <string.h>

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
    const float* hidden, const half* wq_h, const half* wk_h, const half* wv_h, const half* wo_h,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out, float* q, float* k, float* v, float* k_exp, float* v_exp,
    float* attn_out,
    float* q_head, float* k_head, float* v_head, float* o_head,
    int seq_len
);
void mlp_forward(
    const float* hidden, const half* w_gate_h, const half* w_up_h, const half* w_down_h,
    float* out, float* gate_buf, float* up_buf, int seq_len
);
void residual_forward(const float* x, const float* residual, float* out, int n);
void matmul_forward(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_wmma_forward(const float* A, const half* B_half, float* C, int M, int N, int K);
void convert_fp32_to_fp16(const float* d_fp32, half* d_fp16, size_t n);
void attention_decode(
    const float* hidden,
    const half* wq_h, const half* wk_h, const half* wv_h, const half* wo_h,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out, float* q, float* k, float* v, float* attn_out,
    float* k_cache, float* v_cache, int cache_len
);
void quantize_weights(const float* weight, int8_t* q_weight, float* scales, int rows, int cols);
void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales, float* C, int M, int N, int K);
void attention_forward_int8(
    const float* hidden,
    const int8_t* wq_q, const float* wq_s,
    const int8_t* wk_q, const float* wk_s,
    const int8_t* wv_q, const float* wv_s,
    const int8_t* wo_q, const float* wo_s,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out, float* q, float* k, float* v, float* k_exp, float* v_exp,
    float* attn_out, float* q_head, float* k_head, float* v_head, float* o_head,
    int seq_len
);
void mlp_forward_int8(
    const float* hidden,
    const int8_t* w_gate_q, const float* w_gate_s,
    const int8_t* w_up_q,   const float* w_up_s,
    const int8_t* w_down_q, const float* w_down_s,
    float* out, float* gate_buf, float* up_buf, int seq_len
);
void attention_decode_int8(
    const float* hidden,
    const int8_t* wq_q, const float* wq_s,
    const int8_t* wk_q, const float* wk_s,
    const int8_t* wv_q, const float* wv_s,
    const int8_t* wo_q, const float* wo_s,
    const float* q_norm_weight, const float* k_norm_weight,
    float* out, float* q, float* k, float* v, float* attn_out,
    float* k_cache, float* v_cache, int cache_len
);

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
    // fp16 copies for tensor cores
    half* wq_h;
    half* wk_h;
    half* wv_h;
    half* wo_h;
    half* w_gate_h;
    half* w_up_h;
    half* w_down_h;
    // INT8 quantized weights + per-row scales
    int8_t* wq_q;     float* wq_s;
    int8_t* wk_q;     float* wk_s;
    int8_t* wv_q;     float* wv_s;
    int8_t* wo_q;     float* wo_s;
    int8_t* w_gate_q; float* w_gate_s;
    int8_t* w_up_q;   float* w_up_s;
    int8_t* w_down_q; float* w_down_s;
} LayerWeights;

// all model weights
typedef struct {
    float* embed_table;                 // [VOCAB_SIZE, HIDDEN_SIZE]
    LayerWeights layers[NUM_LAYERS];
    float* final_norm;                  // [HIDDEN_SIZE]
    float* lm_head;                     // [VOCAB_SIZE, HIDDEN_SIZE]
    half* lm_head_h;                    // fp16 copy
    int8_t* lm_head_q; float* lm_head_s;
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
            ws->norm_out, lw->wq_h, lw->wk_h, lw->wv_h, lw->wo_h,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->k_exp, ws->v_exp,
            ws->attn_work,
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
        mlp_forward(ws->norm_out, lw->w_gate_h, lw->w_up_h, lw->w_down_h,
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
    matmul_wmma_forward(ws->hidden, model->lm_head_h, ws->logits, seq_len, VOCAB_SIZE, HIDDEN_SIZE);
    debug_save("data/cuda_out/logits.bin", ws->logits, (size_t)seq_len * VOCAB_SIZE);
}

// timed forward pass for benchmarking (no debug saves)
void transformer_forward_bench(
    const int* token_ids,
    const ModelWeights* model,
    Workspace* ws,
    int seq_len,
    FILE* csv
) {
    int hidden_total = seq_len * HIDDEN_SIZE;
    cudaEvent_t t0, t1;
    float ms;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    // total forward
    cudaEvent_t fwd_start, fwd_end;
    CUDA_CHECK(cudaEventCreate(&fwd_start));
    CUDA_CHECK(cudaEventCreate(&fwd_end));
    CUDA_CHECK(cudaEventRecord(fwd_start));

    // embedding
    CUDA_CHECK(cudaEventRecord(t0));
    embed_forward(token_ids, model->embed_table, ws->hidden, seq_len);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "embed,%.4f,\n", ms);

    // accumulate per-component times across all layers
    float total_attn_norm = 0, total_attn = 0, total_ffn_norm = 0, total_mlp = 0, total_residual = 0;

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l];

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        // attn norm
        CUDA_CHECK(cudaEventRecord(t0));
        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_attn_norm += ms;

        // attention
        CUDA_CHECK(cudaEventRecord(t0));
        attention_forward(
            ws->norm_out, lw->wq_h, lw->wk_h, lw->wv_h, lw->wo_h,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->k_exp, ws->v_exp,
            ws->attn_work,
            ws->q_head, ws->k_head, ws->v_head, ws->o_head,
            seq_len
        );
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_attn += ms;

        // residual
        CUDA_CHECK(cudaEventRecord(t0));
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_residual += ms;

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        // ffn norm
        CUDA_CHECK(cudaEventRecord(t0));
        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ffn_norm += ms;

        // mlp
        CUDA_CHECK(cudaEventRecord(t0));
        mlp_forward(ws->norm_out, lw->w_gate_h, lw->w_up_h, lw->w_down_h,
                    ws->attn_out, ws->gate_buf, ws->up_buf, seq_len);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_mlp += ms;

        // residual
        CUDA_CHECK(cudaEventRecord(t0));
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_residual += ms;
    }

    fprintf(csv, "attn_norm (28 layers),%.4f,\n", total_attn_norm);
    fprintf(csv, "attention (28 layers),%.4f,\n", total_attn);
    fprintf(csv, "ffn_norm (28 layers),%.4f,\n", total_ffn_norm);
    fprintf(csv, "mlp (28 layers),%.4f,\n", total_mlp);
    fprintf(csv, "residual (28 layers),%.4f,\n", total_residual);

    // final norm
    CUDA_CHECK(cudaEventRecord(t0));
    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, seq_len, HIDDEN_SIZE);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "final_norm,%.4f,\n", ms);

    // logits
    CUDA_CHECK(cudaEventRecord(t0));
    matmul_wmma_forward(ws->hidden, model->lm_head_h, ws->logits, seq_len, VOCAB_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "lm_head,%.4f,\n", ms);

    // total
    CUDA_CHECK(cudaEventRecord(fwd_end));
    CUDA_CHECK(cudaEventSynchronize(fwd_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, fwd_start, fwd_end));
    fprintf(csv, "total_forward,%.4f,\n", ms);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(fwd_start));
    CUDA_CHECK(cudaEventDestroy(fwd_end));
}

// decode one token with KV cache
// hidden: [1, HIDDEN_SIZE] (single token embedding)
// returns logits in ws->logits [1, VOCAB_SIZE]
void transformer_decode(
    const int* token_id,        // single token on GPU
    const ModelWeights* model,
    Workspace* ws,
    float** k_cache,            // [NUM_LAYERS] array of per-layer K caches
    float** v_cache,            // [NUM_LAYERS] array of per-layer V caches
    int cache_len               // positions already in cache
) {
    // 1) embedding for single token
    embed_forward(token_id, model->embed_table, ws->hidden, 1);

    // 2) transformer layers
    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l];

        // save residual
        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

        // attention norm
        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, 1, HIDDEN_SIZE);

        // decode attention with KV cache
        attention_decode(
            ws->norm_out,
            lw->wq_h, lw->wk_h, lw->wv_h, lw->wo_h,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->attn_work,
            k_cache[l], v_cache[l], cache_len
        );

        // residual after attention
        residual_forward(ws->attn_out, ws->residual, ws->hidden, HIDDEN_SIZE);

        // save residual for MLP
        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

        // ffn norm
        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, 1, HIDDEN_SIZE);

        // MLP
        mlp_forward(ws->norm_out, lw->w_gate_h, lw->w_up_h, lw->w_down_h,
                    ws->attn_out, ws->gate_buf, ws->up_buf, 1);

        // residual after MLP
        residual_forward(ws->attn_out, ws->residual, ws->hidden, HIDDEN_SIZE);
    }

    // 3) final norm
    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, 1, HIDDEN_SIZE);

    // 4) logits
    matmul_wmma_forward(ws->hidden, model->lm_head_h, ws->logits, 1, VOCAB_SIZE, HIDDEN_SIZE);
}

void transformer_forward_int8(
    const int* token_ids,
    const ModelWeights* model,
    Workspace* ws,
    int seq_len
) {
    int hidden_total = seq_len * HIDDEN_SIZE;

    embed_forward(token_ids, model->embed_table, ws->hidden, seq_len);

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l];

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);

        attention_forward_int8(
            ws->norm_out,
            lw->wq_q, lw->wq_s, lw->wk_q, lw->wk_s, lw->wv_q, lw->wv_s, lw->wo_q, lw->wo_s,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->k_exp, ws->v_exp,
            ws->attn_work, ws->q_head, ws->k_head, ws->v_head, ws->o_head,
            seq_len
        );

        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);

        mlp_forward_int8(ws->norm_out,
            lw->w_gate_q, lw->w_gate_s, lw->w_up_q, lw->w_up_s, lw->w_down_q, lw->w_down_s,
            ws->attn_out, ws->gate_buf, ws->up_buf, seq_len);

        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
    }

    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, seq_len, HIDDEN_SIZE);
    matmul_int8_forward(ws->hidden, model->lm_head_q, model->lm_head_s, ws->logits, seq_len, VOCAB_SIZE, HIDDEN_SIZE);
}

void transformer_forward_int8_bench(
    const int* token_ids,
    const ModelWeights* model,
    Workspace* ws,
    int seq_len,
    FILE* csv
) {
    int hidden_total = seq_len * HIDDEN_SIZE;
    cudaEvent_t t0, t1;
    float ms;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    cudaEvent_t fwd_start, fwd_end;
    CUDA_CHECK(cudaEventCreate(&fwd_start));
    CUDA_CHECK(cudaEventCreate(&fwd_end));
    CUDA_CHECK(cudaEventRecord(fwd_start));

    CUDA_CHECK(cudaEventRecord(t0));
    embed_forward(token_ids, model->embed_table, ws->hidden, seq_len);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "embed,%.4f,\n", ms);

    float total_attn_norm = 0, total_attn = 0, total_ffn_norm = 0, total_mlp = 0, total_residual = 0;

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l];

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaEventRecord(t0));
        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_attn_norm += ms;

        CUDA_CHECK(cudaEventRecord(t0));
        attention_forward_int8(
            ws->norm_out,
            lw->wq_q, lw->wq_s, lw->wk_q, lw->wk_s, lw->wv_q, lw->wv_s, lw->wo_q, lw->wo_s,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->k_exp, ws->v_exp,
            ws->attn_work, ws->q_head, ws->k_head, ws->v_head, ws->o_head,
            seq_len
        );
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_attn += ms;

        CUDA_CHECK(cudaEventRecord(t0));
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_residual += ms;

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    hidden_total * sizeof(float), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaEventRecord(t0));
        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, seq_len, HIDDEN_SIZE);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_ffn_norm += ms;

        CUDA_CHECK(cudaEventRecord(t0));
        mlp_forward_int8(ws->norm_out,
            lw->w_gate_q, lw->w_gate_s, lw->w_up_q, lw->w_up_s, lw->w_down_q, lw->w_down_s,
            ws->attn_out, ws->gate_buf, ws->up_buf, seq_len);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_mlp += ms;

        CUDA_CHECK(cudaEventRecord(t0));
        residual_forward(ws->attn_out, ws->residual, ws->hidden, hidden_total);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        total_residual += ms;
    }

    fprintf(csv, "attn_norm (28 layers),%.4f,\n", total_attn_norm);
    fprintf(csv, "attention (28 layers),%.4f,\n", total_attn);
    fprintf(csv, "ffn_norm (28 layers),%.4f,\n", total_ffn_norm);
    fprintf(csv, "mlp (28 layers),%.4f,\n", total_mlp);
    fprintf(csv, "residual (28 layers),%.4f,\n", total_residual);

    CUDA_CHECK(cudaEventRecord(t0));
    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, seq_len, HIDDEN_SIZE);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "final_norm,%.4f,\n", ms);

    CUDA_CHECK(cudaEventRecord(t0));
    matmul_int8_forward(ws->hidden, model->lm_head_q, model->lm_head_s, ws->logits, seq_len, VOCAB_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    fprintf(csv, "lm_head,%.4f,\n", ms);

    CUDA_CHECK(cudaEventRecord(fwd_end));
    CUDA_CHECK(cudaEventSynchronize(fwd_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, fwd_start, fwd_end));
    fprintf(csv, "total_forward,%.4f,\n", ms);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(fwd_start));
    CUDA_CHECK(cudaEventDestroy(fwd_end));
}

void transformer_decode_int8(
    const int* token_id,
    const ModelWeights* model,
    Workspace* ws,
    float** k_cache,
    float** v_cache,
    int cache_len
) {
    embed_forward(token_id, model->embed_table, ws->hidden, 1);

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights* lw = &model->layers[l];

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

        rmsnorm_forward(ws->hidden, lw->attn_norm, ws->norm_out, 1, HIDDEN_SIZE);

        attention_decode_int8(
            ws->norm_out,
            lw->wq_q, lw->wq_s, lw->wk_q, lw->wk_s, lw->wv_q, lw->wv_s, lw->wo_q, lw->wo_s,
            lw->q_norm, lw->k_norm,
            ws->attn_out, ws->q, ws->k, ws->v, ws->attn_work,
            k_cache[l], v_cache[l], cache_len
        );

        residual_forward(ws->attn_out, ws->residual, ws->hidden, HIDDEN_SIZE);

        CUDA_CHECK(cudaMemcpy(ws->residual, ws->hidden,
                    HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));

        rmsnorm_forward(ws->hidden, lw->ffn_norm, ws->norm_out, 1, HIDDEN_SIZE);

        mlp_forward_int8(ws->norm_out,
            lw->w_gate_q, lw->w_gate_s, lw->w_up_q, lw->w_up_s, lw->w_down_q, lw->w_down_s,
            ws->attn_out, ws->gate_buf, ws->up_buf, 1);

        residual_forward(ws->attn_out, ws->residual, ws->hidden, HIDDEN_SIZE);
    }

    rmsnorm_forward(ws->hidden, model->final_norm, ws->hidden, 1, HIDDEN_SIZE);
    matmul_int8_forward(ws->hidden, model->lm_head_q, model->lm_head_s, ws->logits, 1, VOCAB_SIZE, HIDDEN_SIZE);
}