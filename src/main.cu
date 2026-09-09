#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_fp16.h>
#include "tensor.h"
#include "config.h"
#include "kv_cache.h"

void convert_fp32_to_fp16(const float* d_fp32, half* d_fp16, size_t n);
void quantize_weights(const float* weight, int8_t* q_weight, float* scales, int rows, int cols);

// helper: allocate fp16 buffer, convert fp32 weights, return fp16 pointer
half* make_fp16_copy(float* d_fp32, size_t elements) {
    half* d_fp16;
    CUDA_CHECK(cudaMalloc(&d_fp16, elements * sizeof(half)));
    convert_fp32_to_fp16(d_fp32, d_fp16, elements);
    return d_fp16;
}

// helper: allocate int8 buffer + scales, quantize fp32 weights in-place on GPU
void make_int8_copy(float* d_fp32, int8_t** q_out, float** s_out, size_t rows, size_t cols) {
    CUDA_CHECK(cudaMalloc(q_out, rows * cols * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(s_out, rows * sizeof(float)));
    quantize_weights(d_fp32, *q_out, *s_out, (int)rows, (int)cols);
}

// from forward.cu
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

typedef struct {
    float* embed_table;                 // [VOCAB_SIZE, HIDDEN_SIZE]
    LayerWeights layers[NUM_LAYERS];
    float* final_norm;                  // [HIDDEN_SIZE]
    float* lm_head;                     // [VOCAB_SIZE, HIDDEN_SIZE]
    half* lm_head_h;
    int8_t* lm_head_q; float* lm_head_s;
} ModelWeights;

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

void transformer_forward(const int* token_ids, const ModelWeights* model,
                          Workspace* ws, int seq_len);
void transformer_forward_bench(const int* token_ids, const ModelWeights* model,
                                Workspace* ws, int seq_len, FILE* csv);
void transformer_decode(const int* token_id, const ModelWeights* model,
                         Workspace* ws, float** k_cache, float** v_cache, int cache_len);
void transformer_forward_int8(const int* token_ids, const ModelWeights* model,
                               Workspace* ws, int seq_len);
void transformer_forward_int8_bench(const int* token_ids, const ModelWeights* model,
                                     Workspace* ws, int seq_len, FILE* csv);
void transformer_decode_int8(const int* token_id, const ModelWeights* model,
                              Workspace* ws, float** k_cache, float** v_cache, int cache_len);

// load binary file into cpu buffer
float* load_binary_f32(const char* path, size_t expected_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        exit(1);
    }
    float* buf = (float*)malloc(expected_elements * sizeof(float));
    size_t read = fread(buf, sizeof(float), expected_elements, f);
    if (read != expected_elements) {
        fprintf(stderr, "ERROR: %s expected %zu elements, got %zu\n",
                path, expected_elements, read);
        exit(1);
    }
    fclose(f);
    return buf;
}

// load binary file -> gpu, returns device pointer
float* load_weight(const char* path, size_t elements) {
    float* host = load_binary_f32(path, elements);
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host, elements * sizeof(float), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}

// copy gpu buffer to disk
void save_binary_f32(const char* path, const float* dev_ptr, size_t elements) {
    float* host = (float*)malloc(elements * sizeof(float));
    CUDA_CHECK(cudaMemcpy(host, dev_ptr, elements * sizeof(float), cudaMemcpyDeviceToHost));
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot write %s\n", path);
        exit(1);
    }
    fwrite(host, sizeof(float), elements, f);
    fclose(f);
    free(host);
}

// load all qwen3 weights from weights/ dir
void load_model_weights(ModelWeights* model) {
    char path[512];
    printf("Loading weights...\n");

    // embedding table [VOCAB_SIZE, HIDDEN_SIZE]
    model->embed_table = load_weight(
        "weights/model.embed_tokens.weight.bin",
        (size_t)VOCAB_SIZE * HIDDEN_SIZE
    );

    // per-layer weights
    for (int l = 0; l < NUM_LAYERS; l++) {
        LayerWeights* lw = &model->layers[l];

        sprintf(path, "weights/model.layers.%d.input_layernorm.weight.bin", l);
        lw->attn_norm = load_weight(path, HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.post_attention_layernorm.weight.bin", l);
        lw->ffn_norm = load_weight(path, HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.self_attn.q_proj.weight.bin", l);
        lw->wq = load_weight(path, (size_t)NUM_HEADS * HEAD_DIM * HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.self_attn.k_proj.weight.bin", l);
        lw->wk = load_weight(path, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.self_attn.v_proj.weight.bin", l);
        lw->wv = load_weight(path, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.self_attn.o_proj.weight.bin", l);
        lw->wo = load_weight(path, (size_t)HIDDEN_SIZE * NUM_HEADS * HEAD_DIM);

        sprintf(path, "weights/model.layers.%d.self_attn.q_norm.weight.bin", l);
        lw->q_norm = load_weight(path, HEAD_DIM);

        sprintf(path, "weights/model.layers.%d.self_attn.k_norm.weight.bin", l);
        lw->k_norm = load_weight(path, HEAD_DIM);

        sprintf(path, "weights/model.layers.%d.mlp.gate_proj.weight.bin", l);
        lw->w_gate = load_weight(path, (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.mlp.up_proj.weight.bin", l);
        lw->w_up = load_weight(path, (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE);

        sprintf(path, "weights/model.layers.%d.mlp.down_proj.weight.bin", l);
        lw->w_down = load_weight(path, (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE);

        // convert projection weights to fp16 for tensor cores
        lw->wq_h = make_fp16_copy(lw->wq, (size_t)NUM_HEADS * HEAD_DIM * HIDDEN_SIZE);
        lw->wk_h = make_fp16_copy(lw->wk, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE);
        lw->wv_h = make_fp16_copy(lw->wv, (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE);
        lw->wo_h = make_fp16_copy(lw->wo, (size_t)HIDDEN_SIZE * NUM_HEADS * HEAD_DIM);
        lw->w_gate_h = make_fp16_copy(lw->w_gate, (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE);
        lw->w_up_h = make_fp16_copy(lw->w_up, (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE);
        lw->w_down_h = make_fp16_copy(lw->w_down, (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE);

        // quantize to int8
        make_int8_copy(lw->wq, &lw->wq_q, &lw->wq_s, (size_t)NUM_HEADS * HEAD_DIM, HIDDEN_SIZE);
        make_int8_copy(lw->wk, &lw->wk_q, &lw->wk_s, (size_t)NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        make_int8_copy(lw->wv, &lw->wv_q, &lw->wv_s, (size_t)NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        make_int8_copy(lw->wo, &lw->wo_q, &lw->wo_s, (size_t)HIDDEN_SIZE, NUM_HEADS * HEAD_DIM);
        make_int8_copy(lw->w_gate, &lw->w_gate_q, &lw->w_gate_s, (size_t)INTERMEDIATE_SIZE, HIDDEN_SIZE);
        make_int8_copy(lw->w_up,   &lw->w_up_q,   &lw->w_up_s,   (size_t)INTERMEDIATE_SIZE, HIDDEN_SIZE);
        make_int8_copy(lw->w_down, &lw->w_down_q, &lw->w_down_s, (size_t)HIDDEN_SIZE, INTERMEDIATE_SIZE);

        printf("  layer %d loaded\n", l);
    }

    // final norm
    model->final_norm = load_weight("weights/model.norm.weight.bin", HIDDEN_SIZE);

    // lm_head (separate, not tied to embed)
    model->lm_head = load_weight("weights/lm_head.weight.bin",
                                  (size_t)VOCAB_SIZE * HIDDEN_SIZE);
    model->lm_head_h = make_fp16_copy(model->lm_head, (size_t)VOCAB_SIZE * HIDDEN_SIZE);
    make_int8_copy(model->lm_head, &model->lm_head_q, &model->lm_head_s, (size_t)VOCAB_SIZE, HIDDEN_SIZE);

    // memory footprint summary
    size_t fp16_bytes = 0, int8_bytes = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        fp16_bytes += (size_t)NUM_HEADS * HEAD_DIM * HIDDEN_SIZE * 2;     // wq_h
        fp16_bytes += (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * 2;  // wk_h
        fp16_bytes += (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE * 2;  // wv_h
        fp16_bytes += (size_t)HIDDEN_SIZE * NUM_HEADS * HEAD_DIM * 2;     // wo_h
        fp16_bytes += (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE * 2;        // w_gate_h
        fp16_bytes += (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE * 2;        // w_up_h
        fp16_bytes += (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE * 2;        // w_down_h
        int8_bytes += (size_t)NUM_HEADS * HEAD_DIM * HIDDEN_SIZE;
        int8_bytes += (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE;
        int8_bytes += (size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN_SIZE;
        int8_bytes += (size_t)HIDDEN_SIZE * NUM_HEADS * HEAD_DIM;
        int8_bytes += (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;
        int8_bytes += (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;
        int8_bytes += (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE;
    }
    fp16_bytes += (size_t)VOCAB_SIZE * HIDDEN_SIZE * 2;  // lm_head_h
    int8_bytes  += (size_t)VOCAB_SIZE * HIDDEN_SIZE;      // lm_head_q
    printf("Weight memory: FP16=%.1f MB  INT8=%.1f MB\n",
           fp16_bytes / 1e6, int8_bytes / 1e6);

    printf("All weights loaded.\n");
}

// allocate all workspace buffers on gpu
void alloc_workspace(Workspace* ws, int seq_len) {
    size_t S = seq_len;
    size_t H = HIDDEN_SIZE;
    size_t QD = NUM_HEADS * HEAD_DIM;
    size_t KVD = NUM_KV_HEADS * HEAD_DIM;
    size_t I = INTERMEDIATE_SIZE;

    CUDA_CHECK(cudaMalloc(&ws->hidden,    S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->residual,  S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->norm_out,  S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->attn_out,  S * QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->q,         S * QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->k,         S * KVD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->v,         S * KVD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->k_exp,     S * QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->v_exp,     S * QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->attn_work, S * QD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->q_head,   S * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->k_head,   S * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->v_head,   S * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->o_head,   S * HEAD_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->gate_buf,  S * I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->up_buf,    S * I * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->logits,    S * VOCAB_SIZE * sizeof(float)));

    printf("Workspace allocated for seq_len=%d\n", seq_len);
}

void free_workspace(Workspace* ws) {
    cudaFree(ws->hidden);    cudaFree(ws->residual);
    cudaFree(ws->norm_out);  cudaFree(ws->attn_out);
    cudaFree(ws->q);         cudaFree(ws->k);
    cudaFree(ws->v);         cudaFree(ws->k_exp);
    cudaFree(ws->v_exp);     cudaFree(ws->attn_work); cudaFree(ws->q_head);
    cudaFree(ws->k_head);    cudaFree(ws->v_head);
    cudaFree(ws->o_head);    cudaFree(ws->gate_buf);
    cudaFree(ws->up_buf);    cudaFree(ws->logits);
}

int main(int argc, char** argv) {
    // 1) load input tokens
    FILE* f = fopen("data/input_tokens.bin", "rb");
    if (!f) {
        fprintf(stderr, "ERROR: run python/tokenize_input.py first\n");
        return 1;
    }
    int num_tokens;
    fread(&num_tokens, sizeof(int), 1, f);
    int* host_tokens = (int*)malloc(num_tokens * sizeof(int));
    fread(host_tokens, sizeof(int), num_tokens, f);
    fclose(f);
    printf("Loaded %d tokens\n", num_tokens);

    // 2) copy tokens to gpu
    int* dev_tokens;
    CUDA_CHECK(cudaMalloc(&dev_tokens, num_tokens * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dev_tokens, host_tokens, num_tokens * sizeof(int),
                          cudaMemcpyHostToDevice));

    // 3) load model weights
    ModelWeights model;
    load_model_weights(&model);

    // 4) allocate workspace
    Workspace ws;
    alloc_workspace(&ws, num_tokens);

    // 5) parse flags
    int bench_mode = 0;
    int gen_mode = 0;
    int int8_mode = 0;
    int bench_decode = 0;
    int bench_decode_n = 50;
    int max_gen = 50;  // default tokens to generate
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) bench_mode = 1;
        if (strcmp(argv[i], "--generate") == 0) gen_mode = 1;
        if (strcmp(argv[i], "--int8") == 0) int8_mode = 1;
        if (strcmp(argv[i], "--bench-decode") == 0) {
            bench_decode = 1;
            if (i + 1 < argc && argv[i+1][0] != '-') bench_decode_n = atoi(argv[++i]);
        }
        if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_gen = atoi(argv[++i]);
        }
    }

    system("mkdir -p data/cuda_out");

    if (gen_mode) {
        // autoregressive generation with KV cache
        printf("Generating %d tokens...\n", max_gen);

        KVCache cache;
        kv_cache_alloc(&cache);

        int* dev_token;
        CUDA_CHECK(cudaMalloc(&dev_token, sizeof(int)));

        float* host_logits = (float*)malloc((size_t)VOCAB_SIZE * sizeof(float));

        // process prompt tokens
        for (int t = 0; t < num_tokens; t++) {
            CUDA_CHECK(cudaMemcpy(dev_token, &host_tokens[t], sizeof(int), cudaMemcpyHostToDevice));
            if (int8_mode)
                transformer_decode_int8(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            else
                transformer_decode(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            CUDA_CHECK(cudaDeviceSynchronize());
            cache.len++;
        }
        printf("Prompt processed (%d tokens)\n", num_tokens);

        // generate new tokens
        printf("Generated: ");
        for (int g = 0; g < max_gen; g++) {
            CUDA_CHECK(cudaMemcpy(host_logits, ws.logits,
                                  VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            int next_token = 0;
            for (int i = 1; i < VOCAB_SIZE; i++) {
                if (host_logits[i] > host_logits[next_token]) next_token = i;
            }

            if (next_token == 151643 || next_token == 151645) break;

            printf("%d ", next_token);
            fflush(stdout);

            CUDA_CHECK(cudaMemcpy(dev_token, &next_token, sizeof(int), cudaMemcpyHostToDevice));
            if (int8_mode)
                transformer_decode_int8(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            else
                transformer_decode(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            CUDA_CHECK(cudaDeviceSynchronize());
            cache.len++;

            if (cache.len >= MAX_SEQ_LEN) break;
        }
        printf("\n");

        printf("Generation complete. %d total positions in cache.\n", cache.len);

        // decode throughput benchmark
        if (bench_decode) {
            printf("Decode benchmark: %d steps...\n", bench_decode_n);
            CUDA_CHECK(cudaMemcpy(host_logits, ws.logits,
                                  VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            int next_token = 0;
            for (int i = 1; i < VOCAB_SIZE; i++)
                if (host_logits[i] > host_logits[next_token]) next_token = i;
            CUDA_CHECK(cudaMemcpy(dev_token, &next_token, sizeof(int), cudaMemcpyHostToDevice));

            cudaEvent_t d0, d1;
            CUDA_CHECK(cudaEventCreate(&d0));
            CUDA_CHECK(cudaEventCreate(&d1));
            CUDA_CHECK(cudaEventRecord(d0));
            for (int g = 0; g < bench_decode_n; g++) {
                if (int8_mode)
                    transformer_decode_int8(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
                else
                    transformer_decode(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
                cache.len++;
                if (cache.len >= MAX_SEQ_LEN) break;
            }
            CUDA_CHECK(cudaEventRecord(d1));
            CUDA_CHECK(cudaEventSynchronize(d1));
            float decode_ms;
            CUDA_CHECK(cudaEventElapsedTime(&decode_ms, d0, d1));
            float tps = bench_decode_n / (decode_ms / 1000.0f);
            printf("Decode throughput: %.1f tokens/sec (%d tokens in %.1f ms)\n",
                   tps, bench_decode_n, decode_ms);
            CUDA_CHECK(cudaEventDestroy(d0));
            CUDA_CHECK(cudaEventDestroy(d1));
        }

        kv_cache_free(&cache);
        cudaFree(dev_token);
        free(host_logits);

    } else if (bench_mode) {
        // warmup run
        printf("Warmup run...\n");
        if (int8_mode)
            transformer_forward_int8(dev_tokens, &model, &ws, num_tokens);
        else
            transformer_forward(dev_tokens, &model, &ws, num_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        // timed run
        printf("Benchmark run...\n");
        FILE* csv = fopen("data/benchmark.csv", "w");
        fprintf(csv, "kernel,time_ms,gflops\n");
        if (int8_mode)
            transformer_forward_int8_bench(dev_tokens, &model, &ws, num_tokens, csv);
        else
            transformer_forward_bench(dev_tokens, &model, &ws, num_tokens, csv);
        fclose(csv);
        printf("Benchmark saved to data/benchmark.csv\n");

        // decode throughput benchmark
        if (bench_decode) {
            printf("Decode benchmark: %d steps...\n", bench_decode_n);
            KVCache cache;
            kv_cache_alloc(&cache);
            int* dev_token;
            CUDA_CHECK(cudaMalloc(&dev_token, sizeof(int)));
            CUDA_CHECK(cudaMemcpy(dev_token, host_tokens, sizeof(int), cudaMemcpyHostToDevice));

            // warmup decode
            if (int8_mode)
                transformer_decode_int8(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            else
                transformer_decode(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
            cache.len++;
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t d0, d1;
            CUDA_CHECK(cudaEventCreate(&d0));
            CUDA_CHECK(cudaEventCreate(&d1));
            CUDA_CHECK(cudaEventRecord(d0));
            for (int g = 0; g < bench_decode_n; g++) {
                if (int8_mode)
                    transformer_decode_int8(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
                else
                    transformer_decode(dev_token, &model, &ws, cache.k_cache, cache.v_cache, cache.len);
                cache.len++;
                if (cache.len >= MAX_SEQ_LEN) break;
            }
            CUDA_CHECK(cudaEventRecord(d1));
            CUDA_CHECK(cudaEventSynchronize(d1));
            float decode_ms;
            CUDA_CHECK(cudaEventElapsedTime(&decode_ms, d0, d1));
            float tps = bench_decode_n / (decode_ms / 1000.0f);
            printf("Decode throughput: %.1f tokens/sec (%d tokens in %.1f ms)\n",
                   tps, bench_decode_n, decode_ms);
            CUDA_CHECK(cudaEventDestroy(d0));
            CUDA_CHECK(cudaEventDestroy(d1));
            kv_cache_free(&cache);
            cudaFree(dev_token);
        }
    } else {
        printf("Running forward pass...\n");
        if (int8_mode)
            transformer_forward_int8(dev_tokens, &model, &ws, num_tokens);
        else
            transformer_forward(dev_tokens, &model, &ws, num_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Forward pass complete.\n");
        printf("Outputs saved to data/cuda_out/\n");
    }

    // print top predicted token (last position, non-generate mode)
    if (!gen_mode) {
        float* host_logits = (float*)malloc((size_t)VOCAB_SIZE * sizeof(float));
        CUDA_CHECK(cudaMemcpy(host_logits,
                              ws.logits + (num_tokens - 1) * VOCAB_SIZE,
                              VOCAB_SIZE * sizeof(float),
                              cudaMemcpyDeviceToHost));
        int top_token = 0;
        for (int i = 1; i < VOCAB_SIZE; i++) {
            if (host_logits[i] > host_logits[top_token]) top_token = i;
        }
        printf("Top predicted token ID: %d (logit=%.4f)\n",
               top_token, host_logits[top_token]);
        free(host_logits);
    }

    // cleanup
    free(host_tokens);
    cudaFree(dev_tokens);
    free_workspace(&ws);

    return 0;
}
