# CUDA Inference + LoRA Fine-Tuning Engine

A from-scratch, raw-CUDA inference engine and LoRA fine-tuning system for Qwen3 1.7B. **All core kernels written in raw CUDA — no cuBLAS, no cuDNN.**

Benchmarked on **RTX A5000** (24 GB VRAM, sm_86, CUDA 12.6.3).

## Features

| Milestone | What was built |
|-----------|---------------|
| M1/M2 | Full Qwen3 1.7B forward pass in raw CUDA; correctness verified layer-by-layer vs HuggingFace (59/59 PASS) |
| M3 | FP16 WMMA tensor-core matmul; up to **6.28× over FP32** at seq=128 |
| M4 | KV-cached autoregressive decode; **26.4 tokens/sec** (FP16) |
| M5 | INT8 per-channel quantization; **33.8 tokens/sec decode** (1.28× over FP16) |
| M6 | LoRA fine-tuning (rank=16, α=32) on Q/K/V/O projections; full backward + AdamW |

## Architecture

Implements the full Qwen3 1.7B transformer (HIDDEN=2048, 28 layers, 16Q/8KV heads, FFN=6144):

- **Embedding** lookup
- **RMSNorm** with shared-memory reduction + backward
- **Tiled matmul** — FP32 (32×32 tiles), FP16 WMMA (tensor cores), INT8 per-channel
- **Rotary position embeddings** (RoPE, θ=1M) + backward
- **Grouped-query attention** (GQA, QK-norm) with KV cache for decode
- **SwiGLU MLP** (SiLU activation) + backward
- **Cross-entropy loss** — fused softmax + NLL + dlogits kernel
- **LoRA adapters** on Q/K/V/O; init, forward-add, grad accumulation, AdamW

## Benchmarks (RTX A5000)

### M3 — FP16 WMMA forward pass (seq=5, batch prefill)

| Kernel | Time (ms) |
|--------|-----------|
| Attention (all layers) | 26.4 |
| MLP (all layers) | 20.7 |
| lm_head | 2.9 |
| **Total** | **52.2** |

### M3 — Matmul kernel comparison (seq=128, 6144×2048)

| Kernel | TFLOPS | vs FP32 |
|--------|--------|---------|
| FP32 tiled | baseline | 1.00× |
| FP16 WMMA | — | **6.28×** |
| INT8 (decode, seq=1) | — | 2.46–3.69× |

### M4/M5 — Decode throughput

| Mode | tokens/sec |
|------|-----------|
| FP16 KV-cached decode | 26.4 |
| INT8 KV-cached decode | **33.8** (1.28×) |

### M5 — INT8 full forward (seq=5)

INT8 is **slower** than FP16 at prefill (67.7 ms vs 52.2 ms) — expected: INT8 de/requantization overhead dominates at large M. INT8 speedup materializes at decode (seq=1).

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA 12+, sm_80+. On UMIACS cluster: `module load cuda/12.6.3` first.

## Run

```bash
# Correctness verify (M1/M2)
python python/export_weights.py
python python/tokenize_input.py "The capital of France is"
python python/reference_forward.py
./build/llm_engine
python python/verify.py

# Autoregressive generation (FP16)
./build/llm_engine --generate --max-tokens 30

# INT8 generation
./build/llm_engine --int8 --generate --max-tokens 20

# Decode throughput benchmark
./build/llm_engine --generate --bench-decode 100

# LoRA training (M6)
./build/llm_train --lr 1e-4 --steps 100 --seq-len 128 --save-every 50
```

## Full Benchmark Run

```bash
# On cluster (SLURM)
sbatch scripts/bench.slurm

# Or directly (RunPod / any GPU node)
bash scripts/bench_runpod.sh
```
