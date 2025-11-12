#!/usr/bin/env python3
"""
compare_overhead_cpp.py
==============================================================
Analyzes timing CSVs from all compiled TensorRT C++ binaries.

Expected files in ./reports/:
  - run_trt_naive_<timestamp>.csv
  - run_trt_gpu_<timestamp>.csv
  - run_trt_graph_<timestamp>.csv
  - run_trt_pinned_<timestamp>.csv

Each CSV must have: run_id,total_gpu_ms,total_cpu_ms

Outputs:
  • Console summary (mean, 25%, 75%, min, max)
  • Candle-style boxplot (box = IQR, whiskers = min/max, red = median, dot = mean)
    saved as overhead_candle_cpp.png
==============================================================
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

REPORT_DIR = "reports"

# ---------------------------------------------------------------
# Helper: find latest matching file
# ---------------------------------------------------------------
def latest_file(pattern: str):
    files = glob.glob(os.path.join(REPORT_DIR, pattern))
    if not files:
        print(f"⚠️  No file matches: {pattern}")
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

# ---------------------------------------------------------------
# Load CSV safely
# ---------------------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    required = {"total_gpu_ms", "total_cpu_ms"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing expected columns {required}")
    return df

# ---------------------------------------------------------------
# Try to locate newest CSVs for all binaries
# ---------------------------------------------------------------
naive_csv  = latest_file("run_trt_naive_*.csv")
gpu_csv    = latest_file("run_trt_gpu_*.csv")
graph_csv  = latest_file("run_trt_graph_*.csv")
pinned_csv = latest_file("run_trt_pinned_*.csv")

print("==============================================================")
print(" 📂 Automatically selected latest C++ timing CSVs:")
if naive_csv:  print(f"  - Naive baseline       : {os.path.basename(naive_csv)}")
if gpu_csv:    print(f"  - GPU-chained baseline : {os.path.basename(gpu_csv)}")
if graph_csv:  print(f"  - CUDA Graph optimized : {os.path.basename(graph_csv)}")
if pinned_csv: print(f"  - Pinned-memory version: {os.path.basename(pinned_csv)}")
print("==============================================================\n")

# ---------------------------------------------------------------
# Load data dynamically
# ---------------------------------------------------------------
data = {}
if naive_csv:  data["Naive baseline"]        = load_csv(naive_csv)["total_gpu_ms"]
if gpu_csv:    data["GPU-chained"]           = load_csv(gpu_csv)["total_gpu_ms"]
if graph_csv:  data["CUDA Graph optimized"]  = load_csv(graph_csv)["total_gpu_ms"]
if pinned_csv: data["Pinned-memory version"] = load_csv(pinned_csv)["total_gpu_ms"]

if not data:
    raise RuntimeError("No valid CSVs found in ./reports")

# ---------------------------------------------------------------
# Compute descriptive statistics
# ---------------------------------------------------------------
def describe(label, series):
    stats = {
        "label": label,
        "mean": series.mean(),
        "p25":  series.quantile(0.25),
        "p75":  series.quantile(0.75),
        "min":  series.min(),
        "max":  series.max()
    }
    return stats

stats_list = [describe(lbl, s) for lbl, s in data.items()]

# ---------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------
print("==============================================================")
print(" C++ TensorRT GPU Time Statistics (ms)")
print("==============================================================")
print(f"{'Configuration':25s} | {'Mean':>8s} | {'25%':>8s} | {'75%':>8s} | {'Min':>8s} | {'Max':>8s}")
print("-"*75)
for st in stats_list:
    print(f"{st['label']:25s} | {st['mean']:8.3f} | {st['p25']:8.3f} | "
          f"{st['p75']:8.3f} | {st['min']:8.3f} | {st['max']:8.3f}")
print("==============================================================\n")

# ---------------------------------------------------------------
# Candle-style boxplot
# ---------------------------------------------------------------
plt.figure(figsize=(8,6))
labels = list(data.keys())
values = [data[lbl] for lbl in labels]

box = plt.boxplot(
    values,
    labels=labels,
    patch_artist=True,
    showmeans=True,
    showfliers=False,   # hide outlier dots
    meanprops=dict(marker='o', markerfacecolor='black', markersize=6),
    boxprops=dict(facecolor='lightgray', color='black'),
    medianprops=dict(color='red', linewidth=1.5),
    whiskerprops=dict(color='black'),
    capprops=dict(color='black'),
)

plt.title("C++ TensorRT Inference Time Distribution (GPU ms)")
plt.ylabel("Total GPU time (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Optionally limit Y range to reduce top gap (e.g., 95th percentile)
all_vals = np.concatenate([v.values for v in values])
upper_lim = np.percentile(all_vals, 95)
plt.ylim(0, upper_lim * 1.05)

plt.tight_layout()
plt.savefig("overhead_candle_cpp.png", dpi=200)
plt.show()

print("📊 Saved candle-style plot: overhead_candle_cpp.png")
