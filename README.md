# CUDA Inference + LoRA Fine-Tuning Engine

A from-scratch, raw-CUDA inference engine and LoRA fine-tuning system for **Qwen3 1.7B**. No cuBLAS. No cuDNN. Every kernel written by hand.

Benchmarked on **RTX A5000** (24 GB VRAM, sm_86, CUDA 12.6.3).

---

## What was built

| Milestone | Description |
|-----------|-------------|
| M1/M2 | Full Qwen3 1.7B forward pass in raw CUDA — verified layer-by-layer vs HuggingFace (**59/59 PASS**) |
| M3 | FP16 WMMA tensor-core matmul — up to **6.3× speedup** over FP32 at prefill shapes |
| M4 | KV-cached autoregressive decode — **26.4 tokens/sec** (FP16) |
| M5 | INT8 per-channel quantization — **33.8 tokens/sec** decode (**1.28×** over FP16) |
| M6 | LoRA fine-tuning (rank=16, α=32) on Q/K/V/O — full backward + AdamW |

---

## Benchmarks (RTX A5000)

### Decode throughput

![Decode throughput](docs/img/decode_throughput.png)

INT8 decode is **1.28× faster** than FP16 — weight memory halved (3.44 GB → 1.72 GB), bandwidth-bound decode benefits immediately.

---

### Forward pass breakdown — FP16 vs INT8 (seq=5)

![Forward pass breakdown](docs/img/forward_breakdown.png)

At small sequence lengths (prefill seq=5), INT8 is *slower* than FP16 (67.7 ms vs 52.2 ms) — dequantization overhead dominates at small M. INT8 wins only at decode (seq=1).

---

### Matmul kernel throughput — FP32 vs FP16 WMMA vs INT8

![Matmul GFLOPS](docs/img/matmul_gflops.png)

WMMA tensor cores reach **10.4 TFLOPS** on the 128×6144×2048 shape. INT8 underperforms at prefill shapes (the current implementation pays overhead without exploiting the full INT8 pipeline at large M).

---

### FP16 WMMA speedup heatmap

![Speedup heatmap](docs/img/speedup_heatmap.png)

Speedup is nearly flat across projection shapes for a given sequence length — the gain is purely from tensor-core throughput, not memory layout.

---

## Architecture

Qwen3 1.7B: HIDDEN=2048, 28 layers, 16Q/8KV heads (GQA), HEAD_DIM=128, FFN=6144, VOCAB=151936.

**Inference kernels (all raw CUDA):**
- Embedding lookup
- RMSNorm — shared-memory parallel reduction
- Tiled matmul — FP32 (32×32 tiles), FP16 WMMA (tensor cores), INT8 per-channel
- Rotary position embeddings (RoPE, θ=1M)
- Grouped-query attention (QK-norm, KV cache)
- SwiGLU MLP (SiLU activation)
- Softmax — numerically stable row-wise

**Training kernels (M6 LoRA):**
- LoRA forward: `y += scale * B*(A*x)`, rank=16, α=32
- Cross-entropy — fused softmax + NLL + dlogits in one pass
- Full backward: RMSNorm, RoPE, SwiGLU, attention, LoRA grad accumulation
- AdamW optimizer kernel
- `matmul_tn_forward` (A^T @ B) for gradient accumulation

---

## Build & Run

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Export weights (requires HuggingFace Qwen3-1.7B)
python python/export_weights.py

# Correctness verify (M1/M2)
python python/tokenize_input.py "The capital of France is"
python python/reference_forward.py
./build/llm_engine
python python/verify.py

# Autoregressive generation — FP16
./build/llm_engine --generate --max-tokens 30

# Autoregressive generation — INT8
./build/llm_engine --int8 --generate --max-tokens 30

# Decode throughput benchmark (100 steps)
./build/llm_engine --generate --bench-decode 100

# LoRA training (M6)
./build/llm_train --lr 1e-4 --steps 100 --seq-len 128 --save-every 50
```

On UMIACS cluster (`module load cuda/12.6.3` required):
```bash
sbatch scripts/bench.slurm
# or directly:
bash scripts/bench_runpod.sh
```

---

## Weight memory

| Mode | Size |
|------|------|
| FP16 | 3.44 GB |
| INT8 | 1.72 GB |
