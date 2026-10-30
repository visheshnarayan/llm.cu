#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tensor.h"
#include "config.h"
#include "lora.h"

// mirrors main.cu structs (FP32 fields only)
typedef struct {
    float* attn_norm;
    float* ffn_norm;
    float* wq; float* wk; float* wv; float* wo;
    float* q_norm; float* k_norm;
    float* w_gate; float* w_up; float* w_down;
    void* _pad[21];
} LW;
typedef struct {
    float* embed_table;
    LW layers[NUM_LAYERS];
    float* final_norm;
    float* lm_head;
    void* _pad[3];
} MW;

// defined in other .cu files
void transformer_forward_lora(
    const int*, const MW*, const LoraWeights*,
    float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*,
    LoraActs*, int);
void transformer_backward_lora(
    const float*, const MW*, const LoraWeights*, const LoraActs*, LoraGrads*,
    float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*, float*,
    int);
float cross_entropy_forward(const float* logits, const int* targets,
                             float* dlogits, int seq_len);
void adamw_step_lora(LoraWeights*, const LoraGrads*, float, float, float, float, float, int);
void lora_grads_zero(LoraGrads*);
void gpu_zero(float*, int);

// ---- weight loading (same as main.cu) --------------------------------------
static float* load_weight(const char* path, size_t elements) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); exit(1); }
    float* host = (float*)malloc(elements * sizeof(float));
    size_t n = fread(host, sizeof(float), elements, f);
    if (n != elements) { fprintf(stderr, "ERROR: short read %s\n", path); exit(1); }
    fclose(f);
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host, elements * sizeof(float), cudaMemcpyHostToDevice));
    free(host);
    return dev;
}

static void load_model(MW* model) {
    char path[512];
    model->embed_table = load_weight("weights/model.embed_tokens.weight.bin",
                                     (size_t)VOCAB_SIZE * HIDDEN_SIZE);
    for (int l = 0; l < NUM_LAYERS; l++) {
        LW* lw = &model->layers[l];
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
    model->final_norm = load_weight("weights/model.norm.weight.bin", HIDDEN_SIZE);
    model->lm_head    = load_weight("weights/lm_head.weight.bin",
                                    (size_t)VOCAB_SIZE * HIDDEN_SIZE);
}

// ---- training workspace ----------------------------------------------------
typedef struct {
    float* hidden;
    float* residual;
    float* norm_out;
    float* q;  float* k;  float* v;
    float* k_exp; float* v_exp;
    float* attn_out;
    float* gate_buf; float* up_buf;
    float* logits;
    float* dlogits;
    // backward
    float* d_hidden;  float* d_hidden2;
    float* d_norm;
    float* d_q;   float* d_k;   float* d_v;
    float* d_q_exp; float* d_k_exp; float* d_v_exp;
    float* d_pre_o;
    float* d_gate;  float* d_up;
    float* dh_tmp;  float* dx_lora;
} TrainWs;

static void alloc_train_ws(TrainWs* ws, int seq) {
    size_t S = seq;
    int Q = NUM_HEADS * HEAD_DIM, KV = NUM_KV_HEADS * HEAD_DIM, I = INTERMEDIATE_SIZE;
    int H = HIDDEN_SIZE, V = VOCAB_SIZE;
    CUDA_CHECK(cudaMalloc(&ws->hidden,    S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->residual,  S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->norm_out,  S * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->q,         S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->k,         S * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->v,         S * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->k_exp,     S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->v_exp,     S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->attn_out,  S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->gate_buf,  S * I  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->up_buf,    S * I  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->logits,    S * V  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->dlogits,   S * V  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_hidden,  S * H  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_hidden2, S * H  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_norm,    S * H  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_q,       S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_k,       S * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_v,       S * KV * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_q_exp,   S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_k_exp,   S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_v_exp,   S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_pre_o,   S * Q  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_gate,    S * I  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->d_up,      S * I  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->dh_tmp,    S * LORA_RANK * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ws->dx_lora,   S * Q  * sizeof(float)));  // Q_DIM is largest
}

// ---- save / load LoRA weights ----------------------------------------------
static void save_lora(const LoraWeights* lora, int step) {
    char path[256];
    for (int l = 0; l < NUM_LAYERS; l++) {
        const LoraAdapter* adapters[4] = {&lora->wq[l], &lora->wk[l],
                                           &lora->wv[l], &lora->wo[l]};
        const char* names[4] = {"wq", "wk", "wv", "wo"};
        for (int a = 0; a < 4; a++) {
            const LoraAdapter* ad = adapters[a];
            int szA = LORA_RANK * ad->d_in;
            int szB = ad->d_out * LORA_RANK;
            float* hA = (float*)malloc(szA * sizeof(float));
            float* hB = (float*)malloc(szB * sizeof(float));
            CUDA_CHECK(cudaMemcpy(hA, ad->A, szA * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hB, ad->B, szB * sizeof(float), cudaMemcpyDeviceToHost));
            sprintf(path, "weights/lora_step%d_layer%d_%s_A.bin", step, l, names[a]);
            FILE* f = fopen(path, "wb");
            fwrite(hA, sizeof(float), szA, f); fclose(f);
            sprintf(path, "weights/lora_step%d_layer%d_%s_B.bin", step, l, names[a]);
            f = fopen(path, "wb");
            fwrite(hB, sizeof(float), szB, f); fclose(f);
            free(hA); free(hB);
        }
    }
    printf("Saved LoRA weights at step %d\n", step);
}

// ---- main ------------------------------------------------------------------
int main(int argc, char** argv) {
    float lr          = 3e-4f;
    int   steps       = 1000;
    int   seq_len     = 512;
    int   save_every  = 100;
    int   log_every   = 10;
    float beta1       = 0.9f;
    float beta2       = 0.999f;
    float eps_adam    = 1e-8f;
    float wd          = 0.01f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lr")         && i+1<argc) lr         = atof(argv[++i]);
        if (!strcmp(argv[i], "--steps")      && i+1<argc) steps      = atoi(argv[++i]);
        if (!strcmp(argv[i], "--seq-len")    && i+1<argc) seq_len    = atoi(argv[++i]);
        if (!strcmp(argv[i], "--save-every") && i+1<argc) save_every = atoi(argv[++i]);
        if (!strcmp(argv[i], "--log-every")  && i+1<argc) log_every  = atoi(argv[++i]);
    }

    printf("Training config: lr=%.2e steps=%d seq_len=%d\n", lr, steps, seq_len);

    // load training data
    FILE* f = fopen("data/train_tokens.bin", "rb");
    if (!f) {
        fprintf(stderr, "ERROR: data/train_tokens.bin not found. "
                "Tokenize training data first:\n"
                "  python python/tokenize_input.py --train <text_file>\n");
        return 1;
    }
    int num_train_tokens;
    fread(&num_train_tokens, sizeof(int), 1, f);
    int* host_train = (int*)malloc(num_train_tokens * sizeof(int));
    fread(host_train, sizeof(int), num_train_tokens, f);
    fclose(f);
    printf("Loaded %d training tokens\n", num_train_tokens);

    int num_seqs = num_train_tokens / (seq_len + 1);
    if (num_seqs == 0) {
        fprintf(stderr, "ERROR: need at least %d tokens, got %d\n", seq_len + 1, num_train_tokens);
        return 1;
    }
    printf("Training sequences available: %d\n", num_seqs);

    // load frozen base model
    MW model;
    load_model(&model);

    // init LoRA adapters
    LoraWeights lora;
    lora_weights_init(&lora);
    printf("LoRA adapters initialized (rank=%d, alpha=%d, scale=%.1f)\n",
           LORA_RANK, LORA_ALPHA, LORA_SCALE);

    // gradient accumulators
    LoraGrads grads;
    lora_grads_alloc(&grads, &lora);

    // forward activations storage
    LoraActs acts;
    lora_acts_alloc(&acts, seq_len);
    printf("Activation storage allocated for seq_len=%d\n", seq_len);

    // training workspace
    TrainWs ws;
    alloc_train_ws(&ws, seq_len);

    // device token buffer
    int* dev_tokens;
    int* dev_targets;
    CUDA_CHECK(cudaMalloc(&dev_tokens,  seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_targets, seq_len * sizeof(int)));

    system("mkdir -p weights");

    printf("\n=== Training ===\n");
    float running_loss = 0.0f;
    int   step_count   = 0;

    for (int step = 1; step <= steps; step++) {
        // pick a random sequence
        int seq_idx = (step - 1) % num_seqs;
        int* seq_start = host_train + seq_idx * (seq_len + 1);

        // tokens[0..seq_len-1] are inputs; tokens[1..seq_len] are targets
        CUDA_CHECK(cudaMemcpy(dev_tokens,  seq_start,     seq_len * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_targets, seq_start + 1, seq_len * sizeof(int), cudaMemcpyHostToDevice));

        // zero gradients
        lora_grads_zero(&grads);

        // forward pass
        transformer_forward_lora(
            dev_tokens, &model, &lora,
            ws.hidden, ws.residual, ws.norm_out,
            ws.q, ws.k, ws.v, ws.k_exp, ws.v_exp,
            ws.attn_out, ws.gate_buf, ws.up_buf, ws.logits,
            &acts, seq_len
        );

        // loss + dlogits
        float loss = cross_entropy_forward(ws.logits, dev_targets, ws.dlogits, seq_len);

        // backward pass
        transformer_backward_lora(
            ws.dlogits, &model, &lora, &acts, &grads,
            ws.d_hidden, ws.d_hidden2, ws.d_norm,
            ws.d_q, ws.d_k, ws.d_v,
            ws.d_q_exp, ws.d_k_exp, ws.d_v_exp,
            ws.d_pre_o, ws.d_gate, ws.d_up,
            ws.dh_tmp, ws.dx_lora,
            seq_len
        );

        // optimizer step
        adamw_step_lora(&lora, &grads, lr, beta1, beta2, eps_adam, wd, step);
        CUDA_CHECK(cudaDeviceSynchronize());

        running_loss += loss;
        step_count++;

        if (step % log_every == 0) {
            printf("step %4d  loss %.4f  (avg %.4f)\n",
                   step, loss, running_loss / step_count);
            running_loss = 0.0f;
            step_count   = 0;
        }

        if (step % save_every == 0)
            save_lora(&lora, step);
    }

    // final save
    save_lora(&lora, steps);

    printf("\nTraining complete.\n");

    lora_acts_free(&acts, seq_len);
    lora_grads_free(&grads);
    lora_weights_free(&lora);
    free(host_train);
    cudaFree(dev_tokens);
    cudaFree(dev_targets);

    return 0;
}
