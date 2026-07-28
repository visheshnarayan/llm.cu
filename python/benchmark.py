"""Plot benchmark results from data/benchmark.csv"""

import sys
import csv
import os

def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark.csv"
    if not os.path.exists(csv_path):
        print(f"No benchmark data at {csv_path}")
        print("Run ./build/llm_engine --bench first")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data in CSV")
        return

    # print table
    print(f"\n{'Kernel':<30} {'Time (ms)':>12} {'GFLOPS':>10}")
    print("-" * 55)
    for r in rows:
        name = r['kernel']
        ms = float(r['time_ms'])
        gflops = r.get('gflops', '')
        gf_str = f"{float(gflops):.1f}" if gflops else "-"
        print(f"{name:<30} {ms:>12.3f} {gf_str:>10}")

    # try matplotlib for plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = [r['kernel'] for r in rows]
        times = [float(r['time_ms']) for r in rows]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, times, color='#4C72B0')
        ax.set_xlabel('Time (ms)')
        ax.set_title('Kernel Benchmark — Qwen3-1.7B Forward Pass')
        ax.invert_yaxis()
        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{t:.2f}ms', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig('data/benchmark.png', dpi=150)
        print(f"\nPlot saved to data/benchmark.png")
    except ImportError:
        print("\nInstall matplotlib for plots: pip install matplotlib")

if __name__ == "__main__":
    main()
