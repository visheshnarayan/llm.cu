#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

module load cuda/12.6.3

cd ~/benchmark/cuda-llm-engine

echo "=== GPU ==="
nvidia-smi -L
echo ""

echo "=== Build ==="
mkdir -p build data
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j$(nproc) llm_train 2>&1
cd ..
echo ""

echo "=== Tokenize training data ==="
pip install transformers --quiet
python python/tokenize_input.py --train data/train.txt
echo ""

echo "=== Export weights (if not already done) ==="
if [ ! -f weights/model.norm.weight.bin ]; then
    pip install torch safetensors --quiet
    python python/export_weights.py
    cp weights/model.embed_tokens.weight.bin weights/lm_head.weight.bin
fi
echo ""

echo "=== LoRA Training ==="
./build/llm_train \
    --lr 3e-4 \
    --steps 200 \
    --seq-len 128 \
    --log-every 10 \
    --save-every 100

echo ""
echo "=== Done ==="
ls -lh weights/lora_*.bin 2>/dev/null | head -20
