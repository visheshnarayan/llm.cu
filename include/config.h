#ifndef CONFIG_H
#define CONFIG_H

// qwen3 architecture 
#define HIDDEN_SIZE 2048
#define NUM_LAYERS 28
#define NUM_HEADS 16
#define NUM_KV_HEADS 8
#define HEAD_DIM 128
#define INTERMEDIATE_SIZE 6144
#define VOCAB_SIZE 151936
#define MAX_SEQ_LEN 4096

// gqa: how many Q heads share each kv head 
#define KV_HEAD_GROUPS (NUM_HEADS / NUM_KV_HEADS) // = 2

// rope params 
#define ROPE_THETA 1000000.0f

// rmnsNorm epsilon
#define RMS_NORM_EPS 1e-6f

#endif // CONFIG_H
