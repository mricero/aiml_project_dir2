# 🚀 ViT Compression: Efficiency-Accuracy-Latency Trade-off Analysis

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Nightly-EE4C2C.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📌 Project Overview

Vision Transformers (ViTs) achieve state-of-the-art performance across computer vision tasks but suffer from severe computational bottlenecks. This repository contains the codebase and mathematical frameworks for a rigorous analysis of Efficiency-Accuracy-Latency trade-offs on consumer-edge hardware (NVIDIA RTX 5070 Ti Mobile).

The core operation of a ViT, Multi-Head Self-Attention (MHSA), computes interactions between an input sequence of $N$ tokens and requires $\mathcal{O}(N^2 d)$ Floating Point Operations (FLOPs).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V$$

To mitigate this quadratic bottleneck, this project isolates and evaluates three primary neural network compression paradigms, followed by a massive **24-combination Master Interaction Sweep** to map the ultimate Pareto frontier.

---

## 🧮 Compression Modules & Mathematical Frameworks

### 1. Token Reduction: Token Merging (ToMe)

ToMe utilizes bipartite matching to intelligently fuse similar background tokens based on Key-cosine similarity, physically reducing the sequence length. Given two partitioned sets of tokens $A$ and $B$, the cosine similarity is calculated as:

$$S_{ij} = \frac{k_i \cdot k_j}{\|k_i\|_2 \|k_j\|_2} \quad \forall i \in A, j \in B$$

To prevent the Softmax operation from artificially diluting the attention weights of merged tokens, we apply Proportional Attention Scaling using a log-scale bias $\mathbf{s}$:

$$A = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + \log \mathbf{s} \right)$$

---

### 2. Token Reduction: Custom Threshold Masking

Because physically deleting tokens breaks the 2D spatial grid required by hybrid convolutional models (e.g., EfficientFormer), we implemented a custom $L_2$-norm threshold mask. Token importance $I_i$ is proxied via the vector magnitude along the embedding dimension $D$:

$$I_i = \|x_i\|_2 = \sqrt{\sum_{j=1}^D x_{ij}^2}$$

Tokens falling below a dynamic retention threshold $\rho$ are masked by multiplying their embeddings by zero, preserving the tensor shape $(B, N, C)$ for downstream convolutions:

$$X'_{ij} = \begin{cases} X_{ij} & \text{if } I_i \geq \rho \\ 0 & \text{if } I_i < \rho \end{cases}$$

---

### 3. Post-Training Quantization (PTQ)

We utilize `bitsandbytes` to algorithmically truncate 32-bit floating-point weights (FP32).

For **INT8**, Absolute Maximum (AbsMax) scaling dynamically maps weights into the $[-127, 127]$ integer space:

$$S = \frac{127}{\max(|W|)} \implies W_{\text{INT8}} = \text{round}(W \times S)$$

For **INT4**, we utilize NormalFloat (NF4), an optimal data type that maps weights to 16 strictly spaced quantiles of a standard normal distribution $\mathcal{N}(0,1)$:

$$q_i = \Phi^{-1}\left( \frac{i}{16} \right) \quad \text{for } i \in [1, 16]$$

---

### 4. Zero-Shot Structural Attention Head Pruning

Redundant attention heads are surgically excised based on Shannon's Information Entropy. The entropy of the attention distribution $A_{h}$ for head $h$ is:

$$H_{h,i} = -\sum_{j=1}^N A_{h, i, j} \log(A_{h, i, j} + \epsilon)$$

Heads with a high mean entropy exhibit uniform, unfocused attention. We physically slice the $W_Q, W_K, W_V$, and Projection tensors to permanently remove these parameters.

---

## 📂 Repository Structure

```text
📦 aiml_project_dir2
 ┣ 📂 data/                    # (Ignored) ImageNet-1K 50k validation subset (.parquet)
 ┣ 📂 graphs/                  # Auto-generated high-res matplotlib trade-off graphs
 ┣ 📜 baseline_eval.py         # Standard ViT, EfficientFormer, EfficientViT FP32 baselines
 ┣ 📜 custom_drop_sweep.py     # Custom L2-norm threshold token masking hook
 ┣ 📜 entropy_pruning.py       # Shannon entropy calibration and physical QKV slicing
 ┣ 📜 master_sweep.py          # The 24-combination interaction grid search
 ┣ 📜 report.md               # Comprehensive LaTeX academic report
 ┗ 📜 README.md              # Project documentation
```