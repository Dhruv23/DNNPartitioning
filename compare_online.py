#!/usr/bin/env python3
"""
compare_overhead_csv.py
==============================================================
Analyzes timing CSV from the TensorRT C++ binary.

Expected file: 
  - chunk_timing.csv

Each CSV must have: run_id,total_gpu_ms,total_cpu_ms,
and chunk-level GPU timings for EfficientNet and ResNet.

Outputs:
  • Console summary (mean, 25%, 75%, min, max) for total GPU time per run
  • Candle-style boxplots for EfficientNet and ResNet total GPU time per run
    side-by-side, saved as efficientnet_resnet_candle.png
==============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Load the chunk_timing.csv
# ---------------------------------------------------------------
def load_csv(path):
    """Load the chunk_timing.csv file"""
    df = pd.read_csv(path)
    return df

# ---------------------------------------------------------------
# Plotting function for candle-style boxplot
# ---------------------------------------------------------------
def plot_candle_plot(model_names, data, filename):
    """Generates a side-by-side candle-style boxplot for both models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

    for i, ax in enumerate(axes):
        model_name = model_names[i]
        model_data = data[i]
        
        ax.boxplot(
            model_data,
            patch_artist=True,
            showmeans=True,
            showfliers=False,   # hide outlier dots
            meanprops=dict(marker='o', markerfacecolor='black', markersize=6),
            boxprops=dict(facecolor='lightgray', color='black'),
            medianprops=dict(color='red', linewidth=1.5),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
        )
        ax.set_title(f"{model_name} Total GPU Time Per Run (ms)")
        ax.set_ylabel("Total GPU time (ms)")
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        # Dynamic Y-axis max: 0.05 ms above highest mean (green dots)
        means = [np.mean(model_data)]
        max_mean = max(means)

        # Also compute global min for the bottom
        y_min = np.min(model_data)
        y_top = max_mean + 0.02
        y_min = y_min - 0.01
        ax.set_ylim(y_min, y_top)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

    print(f"📊 Saved side-by-side candle-style plot: {filename}")

# ---------------------------------------------------------------
# Compute descriptive statistics for each model
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

# ---------------------------------------------------------------
# Main function to run everything
# ---------------------------------------------------------------
def main():
    # Load data from chunk_timing.csv
    data = load_csv('chunk_timing.csv')

    # EfficientNet and ResNet GPU timing columns
    efficientnet_columns = [f"efficientnet_chunk{i+1}_gpu_ms" for i in range(7)]
    resnet_columns = [f"resnet_chunk{i+1}_gpu_ms" for i in range(8)]

    # ---------------------------------------------------------------
    # Compute the total GPU time per run for EfficientNet
    # ---------------------------------------------------------------
    efficientnet_data = data[efficientnet_columns].sum(axis=1)  # Sum GPU time for each run
    efficientnet_stats = describe("EfficientNet", efficientnet_data)

    # ---------------------------------------------------------------
    # Compute the total GPU time per run for ResNet
    # ---------------------------------------------------------------
    resnet_data = data[resnet_columns].sum(axis=1)  # Sum GPU time for each run
    resnet_stats = describe("ResNet", resnet_data)

    # ---------------------------------------------------------------
    # Print summary table
    # ---------------------------------------------------------------
    print("==============================================================")
    print(" TensorRT GPU Time Statistics (ms)")
    print("==============================================================")
    print(f"{'Configuration':28s} | {'Mean':>8s} | {'25%':>8s} | {'75%':>8s} | {'Min':>8s} | {'Max':>8s}")
    print("-"*80)

    print(f"{efficientnet_stats['label']:28s} | {efficientnet_stats['mean']:8.3f} | "
          f"{efficientnet_stats['p25']:8.3f} | {efficientnet_stats['p75']:8.3f} | "
          f"{efficientnet_stats['min']:8.3f} | {efficientnet_stats['max']:8.3f}")

    print(f"{resnet_stats['label']:28s} | {resnet_stats['mean']:8.3f} | "
          f"{resnet_stats['p25']:8.3f} | {resnet_stats['p75']:8.3f} | "
          f"{resnet_stats['min']:8.3f} | {resnet_stats['max']:8.3f}")

    print("==============================================================\n")

    # ---------------------------------------------------------------
    # Generate side-by-side candle plots for EfficientNet and ResNet
    # ---------------------------------------------------------------
    plot_candle_plot(
        ["EfficientNet", "ResNet"], 
        [efficientnet_data, resnet_data], 
        "efficientnet_resnet_candle.png"
    )

if __name__ == "__main__":
    main()
