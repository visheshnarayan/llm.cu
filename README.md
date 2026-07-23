# CUDA Inference + LoRA Fine-Tuning Engine

A from-scratch, raw-CUDA inference engine and LoRA fine-tuning system for **Qwen3 1.7B**. All core kernels written in raw CUDA — no cuDNN/cuBLAS for the critical path.

## Architecture

Implements the full Qwen3 1.7B transformer in custom CUDA kernels:

- **Embedding** lookup
- **RMSNorm** with shared-memory reduction
- **Tiled matrix multiply** (32×32 shared-memory tiles)
- **Rotary position embeddings** (RoPE, θ=1M)
- **Grouped-query attention** (16 Q heads, 8 KV heads, QK-norm)
- **SwiGLU MLP** (SiLU activation, gated FFN)
- **Softmax** (numerically stable, row-wise)

Correctness verified layer-by-layer against HuggingFace reference outputs.

## Build & Run

_TBD_

## Benchmarks

_TBD_
