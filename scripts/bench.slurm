#!/bin/bash
#SBATCH --job-name=cuda-llm-bench
#SBATCH --partition=class
#SBATCH --account=class
#SBATCH --qos=medium
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=3:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

# load CUDA toolkit (required on UMIACS cluster)
module load cuda/12.6.3

cd ~/benchmark/cuda-llm-engine

echo "=== SLURM job info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
nvidia-smi -L
echo ""

bash scripts/bench_runpod.sh
