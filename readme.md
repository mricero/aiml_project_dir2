# 🚀 ViT Compression: Efficiency-Accuracy-Latency Trade-off Analysis

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-EE4C2C.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📌 Project Overview

Vision Transformers (ViTs) achieve state-of-the-art performance across computer vision tasks but suffer from severe computational bottlenecks.

This repository contains the codebase and mathematical frameworks for a rigorous analysis of Efficiency-Accuracy-Latency trade-offs on consumer-edge hardware (NVIDIA RTX 5070 Ti Mobile).

The core operation of a ViT, Multi-Head Self-Attention (MHSA), computes interactions between an input sequence of $N$ tokens and requires $\mathcal{O}(N^2 d)$ Floating Point Operations (FLOPs).

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

To mitigate this quadratic bottleneck, this project isolates and evaluates three primary neural network compression paradigms, followed by a massive **24-combination Master Interaction Sweep** to map the ultimate Pareto frontier.

---

## 🧮 Compression Modules & Mathematical Frameworks

### 1. Token Reduction: Token Merging (ToMe)

ToMe utilizes bipartite matching to intelligently fuse similar background tokens based on Key-cosine similarity, physically reducing the sequence length.

Given two partitioned sets of tokens A and B, the cosine similarity is calculated as:

S_ij = (k_i · k_j) / (||k_i||_2 ||k_j||_2)  for all i in A, j in B

To prevent the Softmax operation from artificially diluting the attention weights of merged tokens, we apply Proportional Attention Scaling using a log-scale bias s:

A = softmax((QK^T / sqrt(d_k)) + log s)

### 2. Token Reduction: Custom Threshold Masking

Because physically deleting tokens breaks the 2D spatial grid required by hybrid convolutional models (e.g., EfficientFormer), we implemented a custom L2-norm threshold mask. Token importance I_i is proxied via the vector magnitude along the embedding dimension D:

I_i = ||x_i||_2 = sqrt(sum_{j=1}^D x_ij^2)

Tokens falling below a dynamic retention threshold τ are masked by multiplying their embeddings by zero, preserving the tensor shape (B, N, C) for downstream convolutions:

X'_ij = X_ij   if I_i >= τ
X'_ij = 0     if I_i < τ

### 3. Post-Training Quantization (PTQ)

We utilize `bitsandbytes` to algorithmically truncate 32-bit floating-point weights (FP32).

For INT8, Absolute Maximum (AbsMax) scaling dynamically maps weights into the [-127, 127] integer space:

S = 127 / max(|W|)
W_INT8 = round(W × S)

For INT4, we utilize NormalFloat (NF4), an information-theoretically optimal data type that maps weights to 16 strictly spaced quantiles of a standard normal distribution N(0,1):

q_i = Φ^-1(i / 16)   for i in [1, 16]

### 4. Zero-Shot Structural Attention Head Pruning

Redundant attention heads are surgically excised based on Shannon's Information Entropy during a calibration phase. The entropy of the attention distribution A_h for head h is:

H_{h,i} = -sum_{j=1}^N A_{h,i,j} log(A_{h,i,j} + ε)

Heads with a high mean entropy (H̄_h) exhibit uniform, unfocused attention. We physically slice the W_Q, W_K, W_V, and Projection tensors to permanently remove these parameters.

---

## 📂 Repository Structure

aiml_project_dir2/
├── data/                    # (Ignored by Git) ImageNet-1K 50k validation subset (.parquet)
├── graphs/                  # Auto-generated high-res matplotlib trade-off graphs
├── baseline_eval.py         # Standard ViT, EfficientFormer, EfficientViT FP32 baselines
├── custom_drop_sweep.py     # Custom L2-norm threshold token masking hook
├── entropy_pruning.py       # Shannon entropy calibration and physical QKV slicing
├── master_sweep.py          # The 24-combination interaction grid search
├── report.tex               # Comprehensive LaTeX academic report
└── README.md                # Project documentation

---

## ⚙️ Hardware & Environment Setup

This project was engineered and benchmarked on an NVIDIA RTX 5070 Ti Mobile (Blackwell Architecture). Because of heavy monkey-patching and specialized CUDA kernels, the environment must be strictly replicated.

### Installation

    # 1. Create and activate the conda environment
    conda create -n d2_compression python=3.10 -y
    conda activate d2_compression

    # 2. Install PyTorch Nightly (for RTX 5000 series CUDA 12.8 support)
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

    # 3. Install core ML libraries
    pip install timm datasets tqdm matplotlib numpy pandas

    # 4. Install BitsAndBytes (for PTQ)
    pip install bitsandbytes

    # 5. Install Token Merging (Directly from Meta's GitHub)
    pip install git+https://github.com/facebookresearch/ToMe.git

---

## Usage Guide

### Prerequisites

Ensure you have the following installed:
- Anaconda or Miniconda
- NVIDIA Drivers supporting CUDA 12.8 or higher

### Installation

The following commands will set up the required environment:

    # 1. Create and activate the conda environment
    conda create -n d2_compression python=3.10 -y
    conda activate d2_compression

    # 2. Install PyTorch Nightly (for RTX 5000 series CUDA 12.8 support)
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

    # 3. Install core ML libraries
    pip install timm datasets tqdm matplotlib numpy pandas

    # 4. Install BitsAndBytes (for PTQ)
    pip install bitsandbytes

    # 5. Install Token Merging (Directly from Meta's GitHub)
    pip install git+https://github.com/facebookresearch/ToMe.git

### Executing the Code

#### 1. Download the Dataset

The scripts expect a local parquet subset of ImageNet to prevent network I/O bottlenecks. Place your `validation-*.parquet` files inside the `./data/` directory.

#### 2. Run Isolated Sweeps

To test individual techniques and generate their respective Pareto curves:

    python custom_drop_sweep.py
    python entropy_pruning.py

#### 3. Run the Master Interaction Sweep

Executes a 3-dimensional grid search across 24 combinations (Quantization × Pruning × Token Reduction).

Note: Includes aggressive garbage collection, but execution is compute-intensive.

    python master_sweep.py

---

## Key Findings & Discrepancy Analysis

The following section synthesizes the critical insights and addresses the anomalies encountered during our empirical evaluation.

### The Optimal Zero-Shot Stack

The most efficient zero-shot compression stack is **INT4 Quantization + ToMe (r=8)**.

Analysis: The mathematical sequence reduction (O(N^2)) of ToMe perfectly offsets the heavy dequantization/casting latency overhead introduced by `bitsandbytes`. This synergy results in a heavily constrained VRAM footprint (approximately 312 MB) while maintaining a survivable accuracy of 70.02%.

### The Toxicity of Structural Pruning

Zero-shot structural pruning induces catastrophic representation collapse, plunging accuracy to 0.12%.

Analysis: Excising a head physically shifts the numerical variance of the residual stream. This effectively poisons the downstream LayerNorms and MLPs, as they are no longer receiving the expected feature distribution. It mandates a post-pruning fine-tuning or Knowledge Distillation pipeline.

### Masking Compute Overhead

While Custom L2 zero-masking successfully preserved spatial grids, it failed to reduce latency.

Analysis: Standard dense matrix multiplication (C = A × B) does not inherently skip 0.0 operands on GPUs. Without custom C++/CUDA Sparse Tensor kernels, threshold masking yields no actual wall-clock latency acceleration, as the hardware still performs the floating-point operations on the zeroed values.

---

## Conclusion

- Token reduction is the most impactful optimization.
- Quantization provides strong memory savings.
- Structural pruning is unsafe without retraining.
- True latency gains require hardware-aware sparsity.

---

## License

MIT License