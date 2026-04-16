import matplotlib.pyplot as plt
import numpy as np

# 1. Hardcoded Master Sweep Data
data = [
    {"id": "P1", "config": "FP32|P0|None", "acc": 74.59, "lat": 74.90, "vram": 397.25},
    {"id": "P2", "config": "FP32|P0|ToMe_r4", "acc": 74.28, "lat": 76.54, "vram": 394.95},
    {"id": "P3", "config": "FP32|P0|ToMe_r8", "acc": 73.44, "lat": 74.93, "vram": 386.15},
    {"id": "P4", "config": "FP32|P0|Mask_75", "acc": 68.92, "lat": 98.60, "vram": 399.44},
    {"id": "P5", "config": "FP32|P10|None", "acc": 0.12, "lat": 91.75, "vram": 376.46},
    {"id": "P6", "config": "FP32|P10|ToMe_r4", "acc": 0.12, "lat": 84.41, "vram": 393.93},
    {"id": "P7", "config": "FP32|P10|ToMe_r8", "acc": 0.11, "lat": 73.28, "vram": 385.55},
    {"id": "P8", "config": "FP32|P10|Mask_75", "acc": 0.12, "lat": 79.11, "vram": 377.74},
    {"id": "P9", "config": "INT8|P0|None", "acc": 74.57, "lat": 131.83, "vram": 379.92},
    {"id": "P10", "config": "INT8|P0|ToMe_r4", "acc": 74.22, "lat": 129.22, "vram": 371.69},
    {"id": "P11", "config": "INT8|P0|ToMe_r8", "acc": 73.55, "lat": 113.78, "vram": 361.72},
    {"id": "P12", "config": "INT8|P0|Mask_75", "acc": 69.00, "lat": 126.86, "vram": 378.38},
    {"id": "P13", "config": "INT8|P10|None", "acc": 0.11, "lat": 113.48, "vram": 376.23},
    {"id": "P14", "config": "INT8|P10|ToMe_r4", "acc": 0.11, "lat": 114.89, "vram": 369.86},
    {"id": "P15", "config": "INT8|P10|ToMe_r8", "acc": 0.10, "lat": 103.28, "vram": 359.76},
    {"id": "P16", "config": "INT8|P10|Mask_75", "acc": 0.14, "lat": 114.04, "vram": 376.13},
    {"id": "P17", "config": "INT4|P0|None", "acc": 71.12, "lat": 98.64, "vram": 326.05},
    {"id": "P18", "config": "INT4|P0|ToMe_r4", "acc": 70.76, "lat": 95.49, "vram": 321.19},
    {"id": "P19", "config": "INT4|P0|ToMe_r8", "acc": 70.02, "lat": 82.40, "vram": 312.47},
    {"id": "P20", "config": "INT4|P0|Mask_75", "acc": 64.65, "lat": 100.61, "vram": 326.46},
    {"id": "P21", "config": "INT4|P10|None", "acc": 0.12, "lat": 93.87, "vram": 304.77},
    {"id": "P22", "config": "INT4|P10|ToMe_r4", "acc": 0.13, "lat": 89.03, "vram": 320.13},
    {"id": "P23", "config": "INT4|P10|ToMe_r8", "acc": 0.13, "lat": 77.17, "vram": 311.75},
    {"id": "P24", "config": "INT4|P10|Mask_75", "acc": 0.10, "lat": 94.59, "vram": 304.73},
]

# Set up the figure layout
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle('Master Sweep: 24 Combinations (P1 to P24)', fontsize=20, fontweight='bold')

quant_colors = {"FP32": "#1f77b4", "INT8": "#ff7f0e", "INT4": "#2ca02c"}

# ==================================
# Plot 1: Pareto Frontier
# ==================================
ax1 = axes[0]
for d in data:
    q_type = d["config"].split("|")[0]
    ax1.scatter(d["lat"], d["acc"], color=quant_colors[q_type], s=120, edgecolors='black', linewidth=0.5, zorder=3)
    
    # Label the dots with P1, P2, etc. (Offset slightly to avoid overlapping the dot itself)
    ax1.annotate(d["id"], (d["lat"], d["acc"]), xytext=(4, 4), textcoords='offset points', fontsize=10, fontweight='bold')

# Create custom legend for the colors
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=quant_colors["FP32"], markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=quant_colors["INT8"], markersize=10, markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor=quant_colors["INT4"], markersize=10, markeredgecolor='k')]
ax1.legend(custom_lines, ['FP32', 'INT8', 'INT4'], title="Quantization Base", loc="center right", fontsize=12)

ax1.set_title("Pareto Frontier: Accuracy vs Latency", fontsize=16)
ax1.set_xlabel("Average Batch Latency (ms) \u2193", fontsize=14)
ax1.set_ylabel("Top-1 Accuracy (%) \u2191", fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6, zorder=0)
ax1.invert_xaxis()

# ==================================
# Plot 2: Peak VRAM Footprint
# ==================================
ax2 = axes[1]
x_pos = np.arange(len(data))
colors = [quant_colors[d["config"].split("|")[0]] for d in data]
labels = [d["id"] for d in data]
vrams = [d["vram"] for d in data]

ax2.bar(x_pos, vrams, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3)
ax2.set_title("Peak VRAM Consumption", fontsize=16)
ax2.set_ylabel("Memory Allocated (MB) \u2193", fontsize=14)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=11, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# ==================================
# Plot 3: Top-1 Accuracy Impact
# ==================================
ax3 = axes[2]
accuracies = [d["acc"] for d in data]
ax3.bar(x_pos, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3)
ax3.set_title("Top-1 Accuracy Retention", fontsize=16)
ax3.set_ylabel("Accuracy (%) \u2191", fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=11, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax3.set_ylim(0, 100)

# Render and Save
plt.tight_layout()
plt.savefig("master_sweep_labeled.png", dpi=300, bbox_inches='tight')
print("Saved clean labeled image as master_sweep_labeled.png")