"""
Generate all benchmark figures for README.
Run from repo root: python python/plot_results.py results/20261030_144705
"""

import sys
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results/20261030_144705"
OUT = "docs/img"
os.makedirs(OUT, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_FP32  = "#4e79a7"
C_WMMA  = "#f28e2b"
C_INT8  = "#e15759"
C_FP16  = "#59a14f"
C_GRAY  = "#bab0ac"

# ── helpers ───────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def save(fig, name):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {p}")

# =============================================================================
# 1) Forward-pass breakdown  (FP16 vs INT8, stacked bar)
# =============================================================================

def plot_forward_breakdown():
    fp16 = read_csv(f"{RESULTS}/benchmark_fp16.csv")
    int8 = read_csv(f"{RESULTS}/benchmark_int8.csv")

    labels_wanted = ["attention (28 layers)", "mlp (28 layers)", "lm_head",
                     "attn_norm (28 layers)", "ffn_norm (28 layers)", "residual (28 layers)"]
    pretty = {"attention (28 layers)": "Attention",
              "mlp (28 layers)": "MLP",
              "lm_head": "lm_head",
              "attn_norm (28 layers)": "AttnNorm",
              "ffn_norm (28 layers)": "FFN Norm",
              "residual (28 layers)": "Residual"}
    colors = [C_FP16, C_FP32, C_WMMA, C_INT8, C_GRAY, "#76b7b2"]

    fp16_map = {r["kernel"]: float(r["time_ms"]) for r in fp16}
    int8_map = {r["kernel"]: float(r["time_ms"]) for r in int8}

    vals_fp16 = [fp16_map.get(k, 0) for k in labels_wanted]
    vals_int8 = [int8_map.get(k, 0) for k in labels_wanted]
    names = [pretty[k] for k in labels_wanted]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    w = 0.35

    bars1 = ax.bar(x - w/2, vals_fp16, w, color=colors, label="FP16", alpha=0.92, edgecolor="white")
    bars2 = ax.bar(x + w/2, vals_int8, w, color=colors, alpha=0.55, edgecolor="white", hatch="//")

    # totals annotation
    total_fp16 = fp16_map.get("total_forward", sum(vals_fp16))
    total_int8 = int8_map.get("total_forward", sum(vals_int8))
    ax.axhline(0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Forward Pass Breakdown — Qwen3 1.7B (RTX A5000, seq=5)\n"
                 f"FP16 total: {total_fp16:.1f} ms    INT8 total: {total_int8:.1f} ms",
                 fontsize=11)

    fp16_patch = mpatches.Patch(facecolor="#aaaaaa", label=f"FP16  ({total_fp16:.1f} ms total)")
    int8_patch = mpatches.Patch(facecolor="#aaaaaa", alpha=0.5, hatch="//",
                                label=f"INT8  ({total_int8:.1f} ms total)")
    ax.legend(handles=[fp16_patch, int8_patch], fontsize=9, loc="upper right")

    save(fig, "forward_breakdown")

# =============================================================================
# 2) Matmul kernel comparison — GFLOPS grouped bar
# =============================================================================

def plot_matmul():
    txt = open(f"{RESULTS}/bench_matmul.txt").read()

    # parse lines like:  [  1x2048x2048]  FP32: X ms (Y GFLOPS)  WMMA: ...  INT8: ...
    pattern = re.compile(
        r"\[\s*(\d+)x(\d+)x(\d+)\]\s+"
        r"FP32:\s+([\d.]+) ms \(\s*([\d.]+) GFLOPS\)\s+"
        r"WMMA:\s+([\d.]+) ms \(\s*([\d.]+) GFLOPS\)\s+"
        r"INT8:\s+([\d.]+) ms \(\s*([\d.]+) GFLOPS\)"
    )

    rows = []
    for m in pattern.finditer(txt):
        seq, M, K = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rows.append({
            "label": f"{seq}×{M}×{K}",
            "seq": seq,
            "fp32_ms": float(m.group(4)), "fp32_gf": float(m.group(5)),
            "wmma_ms": float(m.group(6)), "wmma_gf": float(m.group(7)),
            "int8_ms": float(m.group(8)), "int8_gf": float(m.group(9)),
        })

    # pick 6 representative rows (decode seq=1 + prefill seq=128)
    shown = [r for r in rows if r["seq"] in (1, 128)][:8]
    labels = [r["label"] for r in shown]
    fp32_gf = [r["fp32_gf"] for r in shown]
    wmma_gf = [r["wmma_gf"] for r in shown]
    int8_gf = [r["int8_gf"] for r in shown]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, fp32_gf, w, color=C_FP32, label="FP32 (tiled)", alpha=0.9)
    ax.bar(x,      wmma_gf, w, color=C_WMMA, label="FP16 WMMA",    alpha=0.9)
    ax.bar(x + w,  int8_gf, w, color=C_INT8, label="INT8",         alpha=0.9)

    # speedup labels on top of WMMA bars
    for i, (f, w_) in enumerate(zip(fp32_gf, wmma_gf)):
        if f > 0:
            ax.text(x[i], w_ + 40, f"{w_/f:.1f}×", ha="center", va="bottom", fontsize=7.5, color=C_WMMA)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("GFLOPS")
    ax.set_title("Matmul Kernel Throughput — FP32 vs FP16 WMMA vs INT8 (RTX A5000)", fontsize=11)
    ax.legend(fontsize=9)
    save(fig, "matmul_gflops")

# =============================================================================
# 3) Decode throughput bar
# =============================================================================

def plot_decode():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    modes = ["FP16\nKV-cached", "INT8\nKV-cached"]
    vals  = [26.4, 33.8]
    colors = [C_FP16, C_INT8]

    bars = ax.bar(modes, vals, color=colors, width=0.45, alpha=0.9, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4,
                f"{v} tok/s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylim(0, 42)
    ax.set_ylabel("Tokens / second")
    ax.set_title("Decode Throughput — Qwen3 1.7B\n(RTX A5000, KV-cached autoregressive)", fontsize=11)
    ax.annotate("1.28×", xy=(1, 33.8), xytext=(1.28, 36.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=10, ha="center")
    save(fig, "decode_throughput")

# =============================================================================
# 4) Matmul speedup heatmap (seq × shape)
# =============================================================================

def plot_speedup_heatmap():
    txt = open(f"{RESULTS}/bench_matmul.txt").read()
    pattern = re.compile(
        r"\[\s*(\d+)x(\d+)x(\d+)\]\s+"
        r"FP32:\s+([\d.]+) ms.*?"
        r"WMMA:\s+([\d.]+) ms"
    )
    data = {}
    for m in pattern.finditer(txt):
        seq = int(m.group(1))
        shape = f"{m.group(2)}×{m.group(3)}"
        fp32  = float(m.group(4))
        wmma  = float(m.group(5))
        data.setdefault(seq, {})[shape] = fp32 / wmma

    seqs   = sorted(data.keys())
    shapes = sorted(next(iter(data.values())).keys())
    matrix = np.array([[data[s].get(sh, 0) for sh in shapes] for s in seqs])

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=1.0)
    plt.colorbar(im, ax=ax, label="Speedup (FP16 WMMA / FP32)")

    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels([f"seq={s}" for s in seqs], fontsize=9)
    ax.set_title("FP16 WMMA Speedup over FP32 by Shape (RTX A5000)", fontsize=11)

    for i in range(len(seqs)):
        for j in range(len(shapes)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}×", ha="center", va="center",
                    fontsize=9, color="black" if v < 4 else "white")

    save(fig, "speedup_heatmap")

# =============================================================================

print("Generating plots...")
plot_forward_breakdown()
plot_matmul()
plot_decode()
plot_speedup_heatmap()
print("Done. Images in", OUT)
