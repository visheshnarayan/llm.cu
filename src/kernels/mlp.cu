#include "tensor.h"
#include "config.h"

// forward dec
void matmul_wmma_forward(const float* A, const half* B_half, float* C, int M, int N, int K);
void silu_forward(const float* gate, const float* up, float* out, int n);

void mlp_forward(
    const float* hidden,        // [seq, HIDDEN_SIZE]
    const half* w_gate_h,       // [INTERMEDIATE_SIZE, HIDDEN_SIZE] fp16
    const half* w_up_h,         // [INTERMEDIATE_SIZE, HIDDEN_SIZE] fp16
    const half* w_down_h,       // [HIDDEN_SIZE, INTERMEDIATE_SIZE] fp16
    float* out,                 // [seq, HIDDEN_SIZE]
    float* gate_buf,            // workspace [seq, INTERMEDIATE_SIZE]
    float* up_buf,              // workspace [seq, INTERMEDIATE_SIZE]
    int seq_len
) {
    int inter_total = seq_len * INTERMEDIATE_SIZE;

    // gate and up projections (tensor cores, fp16 weights)
    matmul_wmma_forward(hidden, w_gate_h, gate_buf, seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, w_up_h, up_buf, seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);

    // fused silu(gate) * up
    silu_forward(gate_buf, up_buf, gate_buf, inter_total);

    // down projection (tensor cores, fp16 weights)
    matmul_wmma_forward(gate_buf, w_down_h, out, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE);
}

void matmul_int8_forward(const float* A, const int8_t* B_q, const float* B_scales, float* C, int M, int N, int K);

void mlp_forward_int8(
    const float* hidden,
    const int8_t* w_gate_q, const float* w_gate_s,
    const int8_t* w_up_q,   const float* w_up_s,
    const int8_t* w_down_q, const float* w_down_s,
    float* out,
    float* gate_buf,
    float* up_buf,
    int seq_len
) {
    int inter_total = seq_len * INTERMEDIATE_SIZE;

    matmul_int8_forward(hidden, w_gate_q, w_gate_s, gate_buf, seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    matmul_int8_forward(hidden, w_up_q,   w_up_s,   up_buf,   seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);

    silu_forward(gate_buf, up_buf, gate_buf, inter_total);

    matmul_int8_forward(gate_buf, w_down_q, w_down_s, out, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE);
}