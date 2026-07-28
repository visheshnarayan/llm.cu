#include "tensor.h"
#include "config.h"

// forward dec
void matmul_wmma_forward(const float* A, const float* B, float* C, int M, int N, int K);
void silu_forward(const float* gate, const float* up, float* out, int n);

void mlp_forward(
    const float* hidden,        // [seq, HIDDEN_SIZE]
    const float* w_gate,        // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const float* w_up,          // [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    const float* w_down,        // [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    float* out,                 // [seq, HIDDEN_SIZE]
    float* gate_buf,            // workspace [seq, INTERMEDIATE_SIZE]
    float* up_buf,              // workspace [seq, INTERMEDIATE_SIZE]
    int seq_len
) {
    int inter_total = seq_len * INTERMEDIATE_SIZE;

    // gate and up projections (tensor cores)
    matmul_wmma_forward(hidden, w_gate, gate_buf, seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    matmul_wmma_forward(hidden, w_up, up_buf, seq_len, INTERMEDIATE_SIZE, HIDDEN_SIZE);

    // fused silu(gate) * up
    silu_forward(gate_buf, up_buf, gate_buf, inter_total);

    // down projection (tensor cores)
    matmul_wmma_forward(gate_buf, w_down, out, seq_len, HIDDEN_SIZE, INTERMEDIATE_SIZE);
}