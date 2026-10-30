#!/bin/bash
# runpod setup + build + test script
# usage: bash scripts/run.sh [verify|bench|generate|bench-matmul|all]

set -e

MODE=${1:-verify}

echo "=== install python deps ==="
pip install torch transformers safetensors numpy matplotlib --quiet

echo "=== build ==="
mkdir -p build data/reference data/cuda_out
cd build && cmake .. && make -j$(nproc) && cd ..

if [ "$MODE" = "verify" ] || [ "$MODE" = "all" ]; then
    echo "=== export weights ==="
    python python/export_weights.py
    cp weights/model.embed_tokens.weight.bin weights/lm_head.weight.bin

    echo "=== tokenize ==="
    python python/tokenize_input.py "The capital of France is"

    echo "=== reference forward ==="
    python python/reference_forward.py

    echo "=== cuda forward ==="
    ./build/llm_engine

    echo "=== verify ==="
    python python/verify.py
fi

if [ "$MODE" = "bench" ] || [ "$MODE" = "all" ]; then
    echo "=== forward pass benchmark ==="
    ./build/llm_engine --bench
    python python/benchmark.py
fi

if [ "$MODE" = "bench-matmul" ] || [ "$MODE" = "all" ]; then
    echo "=== matmul benchmark ==="
    ./build/bench_matmul
    python python/plot_matmul_bench.py
fi

if [ "$MODE" = "generate" ] || [ "$MODE" = "all" ]; then
    echo "=== generate ==="
    python python/tokenize_input.py "The capital of France is"
    ./build/llm_engine --generate --max-tokens 30
fi

if [ "$MODE" = "bench-int8" ] || [ "$MODE" = "all" ]; then
    echo "=== INT8 forward pass benchmark ==="
    ./build/llm_engine --int8 --bench
    cp data/benchmark.csv data/benchmark_int8.csv

    echo "=== FP16 decode throughput ==="
    python python/tokenize_input.py "The capital of France is"
    ./build/llm_engine --generate --bench-decode 100

    echo "=== INT8 decode throughput ==="
    ./build/llm_engine --int8 --generate --bench-decode 100

    echo "=== INT8 generate (correctness check) ==="
    ./build/llm_engine --int8 --generate --max-tokens 20
fi

if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    echo "=== LoRA fine-tuning (M6) ==="
    # tokenize training data if provided
    if [ -f "data/train.txt" ]; then
        python python/tokenize_input.py --train data/train.txt
    fi
    ./build/llm_train --lr 3e-4 --steps 200 --seq-len 256 --save-every 100 --log-every 10
    python python/save_lora.py --list --step 200
fi

echo "=== done ==="
