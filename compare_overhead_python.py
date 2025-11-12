#!/usr/bin/env python3
"""
compare_overhead.py
==============================================================
Automatically finds the latest report CSVs in ./reports and
computes segmentation overheads (RTSS 2025 metric).

Supported CSV patterns:
  - resnet50_*_inference_timing.csv          ← monolithic
  - sequential_inference_gpu_*_chunk_timing.csv ← GPU-chained
  - sequential_inference_pinned_*_chunk_timing.csv ← pinned-memory

Output:
  Prints summary table and saves overhead_comparison.png
==============================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

REPORT_DIR = "reports"

# ---------------------------------------------------------------
# Helper: find latest matching file in reports/
# ---------------------------------------------------------------
def latest_file(pattern: str):
    files = glob.glob(os.path.join(REPORT_DIR, pattern))
    if not files:
        raise FileNotFoundError(f"No file matches: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# ---------------------------------------------------------------
# Load CSV (flexible format detection)
# ---------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    if "total_gpu_ms" in df.columns and "total_cpu_ms" in df.columns:
        df = df[["total_gpu_ms", "total_cpu_ms"]].copy()
    elif "gpu_time_ms" in df.columns and "cpu_time_ms" in df.columns:
        df = df.rename(columns={"gpu_time_ms": "total_gpu_ms",
                                "cpu_time_ms": "total_cpu_ms"})
    else:
        raise ValueError(f"{path} missing expected timing columns")
    return df

# ---------------------------------------------------------------
# Summaries and overhead computation
# ---------------------------------------------------------------
def summarize(label, df):
    gpu_mean = df["total_gpu_ms"].mean()
    gpu_std  = df["total_gpu_ms"].std()
    cpu_mean = df["total_cpu_ms"].mean()
    cpu_std  = df["total_cpu_ms"].std()
    return {
        "label": label,
        "gpu_mean": gpu_mean,
        "gpu_std": gpu_std,
        "cpu_mean": cpu_mean,
        "cpu_std": cpu_std
    }

def compute_overhead(base, test):
    return (test["gpu_mean"] - base["gpu_mean"]) / base["gpu_mean"] * 100.0

# ---------------------------------------------------------------
# Auto-locate most recent CSVs
# ---------------------------------------------------------------
mono_csv   = latest_file("resnet50_*_inference_timing.csv")
gpu_csv    = latest_file("sequential_inference_gpu_*_chunk_timing.csv")
pinned_csv = latest_file("sequential_inference_pinned_*_chunk_timing.csv")

print("==============================================================")
print(" 📂 Automatically selected latest CSVs:")
print(f"  - Monolithic baseline : {os.path.basename(mono_csv)}")
print(f"  - GPU-chained         : {os.path.basename(gpu_csv)}")
print(f"  - Pinned-memory       : {os.path.basename(pinned_csv)}")
print("==============================================================")

# ---------------------------------------------------------------
# Load and analyze data
# ---------------------------------------------------------------
mono_df   = load_csv(mono_csv)
gpu_df    = load_csv(gpu_csv)
pinned_df = load_csv(pinned_csv)

mono_stats   = summarize("Monolithic (non-split)", mono_df)
gpu_stats    = summarize("Sequential GPU-chained", gpu_df)
pinned_stats = summarize("Sequential pinned-memory", pinned_df)

gpu_overhead    = compute_overhead(mono_stats, gpu_stats)
pinned_overhead = compute_overhead(mono_stats, pinned_stats)

# ---------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------
print("\n==============================================================")
print(" Segmentation Overhead Comparison (RTSS 2025 metric)")
print("==============================================================")
print(f"{'Configuration':30s} | {'GPU mean (ms)':>12s} | {'CPU mean (ms)':>12s} | {'Overhead % vs mono':>18s}")
print("-"*85)
print(f"{mono_stats['label']:30s} | {mono_stats['gpu_mean']:12.3f} | {mono_stats['cpu_mean']:12.3f} | {'—':>18s}")
print(f"{gpu_stats['label']:30s}  | {gpu_stats['gpu_mean']:12.3f} | {gpu_stats['cpu_mean']:12.3f} | {gpu_overhead:18.2f}")
print(f"{pinned_stats['label']:30s}  | {pinned_stats['gpu_mean']:12.3f} | {pinned_stats['cpu_mean']:12.3f} | {pinned_overhead:18.2f}")
print("==============================================================\n")

# ---------------------------------------------------------------
# Optional bar plot
# ---------------------------------------------------------------
labels = ["Monolithic", "GPU-chained", "Pinned"]
gpu_means = [mono_stats["gpu_mean"], gpu_stats["gpu_mean"], pinned_stats["gpu_mean"]]

plt.figure(figsize=(6,4))
plt.bar(labels, gpu_means, color=["gray","green","orange"])
plt.ylabel("Mean total GPU time (ms)")
plt.title("Sequential Segmentation Overhead")
for i,v in enumerate(gpu_means):
    plt.text(i, v+0.01, f"{v:.3f} ms", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig("overhead_comparison.png", dpi=200)
plt.show()

print("📊 Saved plot: overhead_comparison.png")
