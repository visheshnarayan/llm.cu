#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h> 
#include "utils.h"

// max dims we need (batch seq heads head_dim) 
#define MAX_DIMS 4

typedef struct { 
    float* data;    // device ptr
    int shape[MAX_DIMS];
    int ndim; 
    size_t size;    // total elements 
} Tensor; 

// compute total elements from shape 
static inline size_t tensor_elements(const int* shape, int ndim) { 
    size_t n = 1; 
    for (int i = 0 ; i < ndim ; i++ ) {
        n *= shape[i]; 
    }
    return n; 
}

// allocate tensor on GPU 
static inline Tensor tensor_alloc(const int* shape, int ndim) { 
    Tensor t; 
    t.ndim = ndim; 
    t.size = tensor_elements(shape, ndim); 
    for (int i = 0 ; i < ndim ; i++ ) {
        t.shape[i] = shape[i]; 
    }
    CUDA_CHECK(cudaMalloc(&t.data, t.size * sizeof(float))); 
    return t; 
}

// free a tensor's GPU mem
static inline void tensor_free(Tensor* t) { 
    if (t->data) { 
        CUDA_CHECK(cudaFree(t->data)); 
        t->data = NULL; 
    }
    t->size = 0; 
}

#endif // TENSOR_H 