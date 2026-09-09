"""Plot matmul benchmark: FP32 vs WMMA(FP16) vs INT8"""

import csv
import os
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/bench_matmul.csv"
    if not os.path.exists(path):
        print(f"No data at {path}. Run ./build/bench_matmul first")
        return

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"\n{'Shape':<22} {'FP32 ms':>10} {'WMMA ms':>10} {'INT8 ms':>10} {'WMMA speedup':>14} {'INT8 speedup':>14}")
    print("-" * 85)
    for r in rows:
        shape = f"{r['M']}x{r['N']}x{r['K']}"
        fp32 = float(r['fp32_ms'])
        wmma = float(r['wmma_ms'])
        int8 = float(r['int8_ms'])
        ws = f"{fp32/wmma:.2f}x" if wmma > 0 else "-"
        i8s = f"{fp32/int8:.2f}x" if int8 > 0 else "-"
        print(f"{shape:<22} {fp32:>10.3f} {wmma:>10.3f} {int8:>10.3f} {ws:>14} {i8s:>14}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np

        shapes = [f"{r['M']}x{r['N']}x{r['K']}" for r in rows]
        fp32_gf = [float(r['fp32_gflops']) for r in rows]
        wmma_gf = [float(r['wmma_gflops']) for r in rows]
        int8_gf = [float(r['int8_gflops']) for r in rows]

        x = np.arange(len(shapes))
        w = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w, fp32_gf, w, label='FP32 tiled', color='#4C72B0')
        ax.bar(x, wmma_gf, w, label='WMMA FP16 TC', color='#55A868')
        ax.bar(x + w, int8_gf, w, label='INT8', color='#C44E52')
        ax.set_xticks(x)
        ax.set_xticklabels(shapes, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('GFLOPS')
        ax.set_title('Matmul Performance: FP32 vs WMMA(FP16) vs INT8 — Qwen3 Shapes')
        ax.legend()
        plt.tight_layout()
        plt.savefig('data/bench_matmul.png', dpi=150)
        print(f"\nPlot saved to data/bench_matmul.png")

    except ImportError:
        print("\nInstall matplotlib for plots: pip install matplotlib")

if __name__ == "__main__":
    main()
