#!/bin/bash
# runpod setup + build + test script
# usage: bash scripts/run.sh

set -e

echo "=== 1) install python deps ==="
pip install torch transformers safetensors numpy --quiet

echo "=== 2) build cuda engine ==="
mkdir -p build
cd build
cmake ..
make -j$(nproc)
cd ..

echo "=== 3) export weights ==="
python python/export_weights.py

echo "=== 4) tokenize test prompt ==="
python python/tokenize_input.py "The capital of France is"

echo "=== 5) generate reference outputs ==="
python python/reference_forward.py

echo "=== 6) run cuda forward pass ==="
./build/llm_engine

echo "=== 7) verify against reference ==="
python python/verify.py

echo "=== done ==="
