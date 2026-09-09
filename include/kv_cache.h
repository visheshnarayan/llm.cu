#ifndef KV_CACHE_H
#define KV_CACHE_H

#include "config.h"
#include "utils.h"

// per-layer KV cache: stores all past K and V vectors
// k_cache[pos, NUM_KV_HEADS, HEAD_DIM]
// v_cache[pos, NUM_KV_HEADS, HEAD_DIM]
typedef struct {
    float* k_cache[NUM_LAYERS];  // [MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM]
    float* v_cache[NUM_LAYERS];  // [MAX_SEQ_LEN, NUM_KV_HEADS * HEAD_DIM]
    int len;                      // current number of cached positions
} KVCache;

static inline void kv_cache_alloc(KVCache* cache) {
    size_t kv_size = (size_t)MAX_SEQ_LEN * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    for (int l = 0; l < NUM_LAYERS; l++) {
        CUDA_CHECK(cudaMalloc(&cache->k_cache[l], kv_size));
        CUDA_CHECK(cudaMalloc(&cache->v_cache[l], kv_size));
    }
    cache->len = 0;
}

static inline void kv_cache_free(KVCache* cache) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        cudaFree(cache->k_cache[l]);
        cudaFree(cache->v_cache[l]);
    }
    cache->len = 0;
}

#endif // KV_CACHE_H
