#!/bin/bash
# ===========================================================================
# RunPod all-in-one benchmark script
# Runs every benchmark, captures all output, collects artifacts into results/.
#
# SCP to pod:
#   scp -r . root@<pod-ip>:<ssh-port>:/workspace/cuda-llm-engine/
#
# Run on pod:
#   cd /workspace/cuda-llm-engine && bash scripts/bench_runpod.sh
#
# SCP results back:
#   scp -P <ssh-port> -r root@<pod-ip>:/workspace/cuda-llm-engine/results/ ./results/
# ===========================================================================
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS="results/${TIMESTAMP}"
LOG="${RESULTS}/run.log"

mkdir -p "${RESULTS}"

# tee everything to log
exec > >(tee -a "${LOG}") 2>&1

echo "========================================================"
echo " cuda-llm-engine benchmark run"
echo " timestamp: ${TIMESTAMP}"
echo " host:      $(hostname)"
echo "========================================================"
echo ""

# ── 0) GPU info ──────────────────────────────────────────────────────────────
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader || true
echo ""

# ── 1) Install Python deps ────────────────────────────────────────────────────
echo "=== Python deps ==="
pip install torch transformers safetensors numpy matplotlib --quiet
echo "done"
echo ""

# ── 2) Build ──────────────────────────────────────────────────────────────────
echo "=== Build ==="
mkdir -p build data/reference data/cuda_out
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
make -j$(nproc) 2>&1
cd ..
echo ""

# ── 3) Export weights + tokenize ──────────────────────────────────────────────
echo "=== Export weights ==="
python python/export_weights.py
cp weights/model.embed_tokens.weight.bin weights/lm_head.weight.bin
echo ""

echo "=== Tokenize ==="
python python/tokenize_input.py "The capital of France is"
echo ""

# ── 4) M1/M2: Correctness verify ─────────────────────────────────────────────
echo "========================================================"
echo " M1/M2 CORRECTNESS VERIFY"
echo "========================================================"
python python/reference_forward.py
./build/llm_engine
python python/verify.py
echo ""

# ── 5) M3: FP16/WMMA forward benchmark ───────────────────────────────────────
echo "========================================================"
echo " M3: TENSOR-CORE (FP16/WMMA) FORWARD BENCHMARK"
echo "========================================================"
./build/llm_engine --bench
cp data/benchmark.csv "${RESULTS}/benchmark_fp16.csv"
python python/benchmark.py data/benchmark.csv
cp data/benchmark.png "${RESULTS}/benchmark_fp16.png" 2>/dev/null || true
echo ""

# ── 6) M3: Standalone matmul benchmark (FP32 vs WMMA vs INT8) ────────────────
echo "========================================================"
echo " M3: MATMUL KERNEL BENCHMARK (FP32 vs WMMA vs INT8)"
echo "========================================================"
./build/bench_matmul 2>&1 | tee "${RESULTS}/bench_matmul.txt"
if ls data/matmul_bench*.csv 2>/dev/null; then
    cp data/matmul_bench*.csv "${RESULTS}/"
fi
python python/plot_matmul_bench.py 2>/dev/null || true
cp data/matmul_bench*.png "${RESULTS}/" 2>/dev/null || true
echo ""

# ── 7) M4: KV-cached decode + throughput (FP16) ──────────────────────────────
echo "========================================================"
echo " M4: KV-CACHED DECODE — FP16"
echo "========================================================"
echo "--- Autoregressive generation (correctness) ---"
./build/llm_engine --generate --max-tokens 30

echo ""
echo "--- Decode throughput benchmark (100 steps) ---"
./build/llm_engine --generate --bench-decode 100 | tee "${RESULTS}/decode_fp16.txt"
echo ""

# ── 8) M5: INT8 forward + decode benchmarks ───────────────────────────────────
echo "========================================================"
echo " M5: INT8 QUANTIZED FORWARD BENCHMARK"
echo "========================================================"
./build/llm_engine --int8 --bench
cp data/benchmark.csv "${RESULTS}/benchmark_int8.csv"
python python/benchmark.py data/benchmark.csv
cp data/benchmark.png "${RESULTS}/benchmark_int8.png" 2>/dev/null || true
echo ""

echo "========================================================"
echo " M5: KV-CACHED DECODE — INT8"
echo "========================================================"
./build/llm_engine --int8 --generate --bench-decode 100 | tee "${RESULTS}/decode_int8.txt"
echo ""

echo "--- INT8 generation (correctness) ---"
./build/llm_engine --int8 --generate --max-tokens 20
echo ""

# ── 9) Summary table ─────────────────────────────────────────────────────────
echo "========================================================"
echo " SUMMARY"
echo "========================================================"

RESULTS_DIR="${RESULTS}" python3 - <<'PYEOF'
import csv, os, re

rdir = os.environ["RESULTS_DIR"]

def read_csv(path):
    if not os.path.exists(path): return []
    with open(path) as f:
        return list(csv.DictReader(f))

fp16 = read_csv(f"{rdir}/benchmark_fp16.csv")
int8 = read_csv(f"{rdir}/benchmark_int8.csv")

if fp16:
    print(f"\n{'Kernel':<30} {'FP16 (ms)':>12} {'INT8 (ms)':>12} {'Speedup':>10}")
    print("-" * 67)
    int8_map = {r['kernel']: float(r['time_ms']) for r in int8}
    for r in fp16:
        k   = r['kernel']
        t16 = float(r['time_ms'])
        t8  = int8_map.get(k)
        sp  = f"{t16/t8:.2f}x" if t8 else "-"
        t8s = f"{t8:.3f}" if t8 else "-"
        print(f"{k:<30} {t16:>12.3f} {t8s:>12} {sp:>10}")

for fname, label in [("decode_fp16.txt", "FP16 decode"), ("decode_int8.txt", "INT8 decode")]:
    path = f"{rdir}/{fname}"
    if os.path.exists(path):
        txt = open(path).read()
        m = re.search(r"([\d.]+) tokens/sec", txt)
        if m: print(f"\n{label}: {m.group(1)} tokens/sec")
PYEOF

echo ""
echo "All artifacts saved to: ${RESULTS}/"
ls -lh "${RESULTS}/"
echo ""
echo "========================================================"
echo " Done. SCP results back with:"
echo "   scp -P <port> -r root@<ip>:$(pwd)/${RESULTS}/ ./results/"
echo "========================================================"
