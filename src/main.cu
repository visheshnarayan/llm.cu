#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "config.h"

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
} LayerWeights;

typedef struct {
    float* embed_table;                 // [VOCAB_SIZE, HIDDEN_SIZE]
    LayerWeights layers[NUM_LAYERS];
    float* final_norm;                  // [HIDDEN_SIZE]
    float* lm_head;                     // [VOCAB_SIZE, HIDDEN_SIZE]
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

        printf("  layer %d loaded\n", l);
    }

    // final norm
    model->final_norm = load_weight("weights/model.norm.weight.bin", HIDDEN_SIZE);

    // lm_head (separate, not tied to embed)
    model->lm_head = load_weight("weights/lm_head.weight.bin",
                                  (size_t)VOCAB_SIZE * HIDDEN_SIZE);

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

    // 5) check for --bench flag
    int bench_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) bench_mode = 1;
    }

    system("mkdir -p data/cuda_out");

    if (bench_mode) {
        // warmup run
        printf("Warmup run...\n");
        transformer_forward(dev_tokens, &model, &ws, num_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());

        // timed run
        printf("Benchmark run...\n");
        FILE* csv = fopen("data/benchmark.csv", "w");
        fprintf(csv, "kernel,time_ms,gflops\n");
        transformer_forward_bench(dev_tokens, &model, &ws, num_tokens, csv);
        fclose(csv);
        printf("Benchmark saved to data/benchmark.csv\n");
    } else {
        printf("Running forward pass...\n");
        transformer_forward(dev_tokens, &model, &ws, num_tokens);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Forward pass complete.\n");
        printf("Outputs saved to data/cuda_out/\n");
    }

    // 7) print top predicted token (last position)
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

    // cleanup
    free(host_tokens);
    free(host_logits);
    cudaFree(dev_tokens);
    free_workspace(&ws);

    return 0;
}
