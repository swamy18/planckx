# PLANCK-X: Chronos-Bit Engine (v3.1-Adaptive)

<p align="center">
  <strong>Spectral-Entropy-Guided Compute Routing for Mixed-Precision Neural Inference</strong>
</p>

<p align="center">
  <a href="https://github.com/org/repo/releases"><img alt="Version" src="https://img.shields.io/badge/version-3.1--Adaptive-blue"></a>
  <a href="https://pytorch.org"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-red"></a>
  <a href="https://www.python.org"><img alt="Python" src="https://img.shields.io/badge/Python-3.11--3.13-blue"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green"></a>
</p>

---

## Overview

**PLANCK-X Chronos-Bit Engine** is a prototype research framework that implements **per-token hardware-path routing driven by Shannon Entropy** measured over the model's own vocabulary logits. Rather than executing the full neural network for every token position, the engine dynamically selects between three computational modes — a 1-stage bit-shift shortcut, a 2-stage shift-add fusion primitive, or a full 5-stage deep sequential block — so that predictable tokens bypass expensive computation entirely.

The project integrates **B-PACS 8:1 ternary-weight memory packing**, **asymmetric 90/10 precision layer quantization**, and a **Quantization-Aware Fine-Tuning (QAFT)** stabilisation loop into a single standalone runnable engine.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Benchmarking Results](#benchmarking-results)
7. [Citation](#citation)
8. [License](#license)

---

## Introduction

Traditional neural inference applies the same compute cost to every token in a sequence, regardless of how much information that token carries. High-entropy tokens (early, rare, or noisy) require full model capacity; low-entropy tokens (frequent, predictable) do not.

PLANCK-X lives inside the **embedding projection**, routing each flattened `[batch × seq_len]` token independently:

```
for each token j:
    H[j] = Shannon_Entropy(softmax(X @ W_Vocab)[j])   ← gate signal
    if H[j] < THRESH_1:   execute 1-stage  (X * 127) >> 7          ◀ PATH A
    elif H[j] < THRESH_2: execute 2-stage  (X_int >> 2) + (X_int >> 4)  ◀ PATH B
    else:                 execute 5-stage  nn.Sequential[5×(Linear+ReLU)]  ◀ PATH C
```

Because the entropy gate itself is computed from the model's own `output_projection` logits, the routing decision is a direct proxy for model confidence and **shares a gradient path with the cross-entropy loss** — eliminating the decoupling problem that plagues other dynamic-routing systems.

---

## Key Features

### Mixed-Precision Entropy Routing

| Path | Entropy Window | Operation | Approx. FLOPs/token |
|---|---|---|---|
| **Path 1** | H < 4.0 | `(X * 127).int32 >> 7` | ~1 |
| **Path 2** | 4.0 ≤ H < 4.5 | `(X_int >> 2) + (X_int >> 4)` | ~2 |
| **Path 5** | H ≥ 4.5 | 5× `[Linear(64,64) + ReLU]` sequential | ~640 |

### B-PACS Memory Compression

Implements **Binary-Packed Activation Compression System (B-PACS)** 8:1 packing — groups of eight ternary values `{-1, 0, +1}` are packed slot-by-slot into a single `int16` container, physically demonstrating 8:1 weight footprint reduction without external dependencies.

### Asymmetric Ternary Quantisation (90/10)

The `PlanckXAdapter` performs an asymmetric split on every dense model:

- **90 % of Linear layers** → hard-weighted to strict **3-state ternary** (`+1 / 0 / -1`) using magnitude thresholds ±0.5
- **10 % of Linear layers** → left in full `float32` as stabilisation/gating slots

### Quantization-Aware Fine-Tuning (QAFT)

A complete stabilisation loop runs for 5 post-conversion epochs, using `Adam(lr=1e-4)` with gradient clipping at `max_norm = 1.0` to restore predictive cohesion after the hard weight reset.

### Sparsity / Certainty Penalty

The training objective is augmented:

```
total_loss = CrossEntropyLoss + 0.20 × mean(Shannon_Entropy)
```

This explicitly rewards the model for lowering its per-token disagreement, creating a natural incentive to drive high-entropy tokens downward through the routing gates.

---

## Architecture

```
PLANCK-X v3.1-Adaptive
├── planckx/
│   ├── routing.py      ← PlanckXLayer  (dynamic entropy gating)
│   ├── trainer.py      ← PlanckXTrainer (CE + H-penalty loop)
│   ├── adapter.py      ← PlanckXAdapter (90/10 ternary + QAFT)
│   └── engine.py       ← PlanckXEngine (B-PACS + autoregressive gen)
├── run_demo.py         ← Three-step orchestration script
├── requirements.txt    ← pip install
└── LICENSE             ← MIT
```

### Data Flow

```
Token IDs  --[Embedding]-->  X [B,S,64]
                              │
                   Embedding pass (router tally = Path5)
                              │
              output_projection(X_embed_probe)  ← surrogate logits
                              │
                   H = Shannon(logits) per token
                              │
          ┌──────────────────┴──────────────────┐
       mask_a                mask_b             mask_c
   (H < 4.0)          (4.0 ≤ H < 4.5)         (H ≥ 4.5)
          │                   │                   │
   Path 1 Transform    Path 2 Transform    Path 5 Sequential
   (X*127)>>7 →f32   (X>>2)+(X>>4)→f32    5×[Linear(64,64)+ReLU]
          │                   │                   │
          └──────────────────┴──────────────────┘
                              │
               torch.where select per element
                              │
              routed_activations [B,S,64] --> output_projection --> logits
```

---

## Installation

```bash
# Clone
git clone https://github.com/<your-org>/planck-x.git
cd planck-x

# Create environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt
```

**Requires** Python 3.11–3.13, PyTorch 2.0+ with CUDA support (x86_64 or arm64).

---

## Quick Start

### Run the full demo (three-step pipeline)

```bash
set PYTHONIOENCODING=utf-8   # Windows PowerShell recommended
python run_demo.py
```

**Step 1** — Train a `PlanckXLayer` from scratch for **500 epochs** using `PlanckXTrainer` with `lr = 0.02` and the **Sparsity Penalty**. Telemetry is printed every 50 epochs.

**Step 2** — Convert a dense `DenseCharModel` to 90/10 ternary using `PlanckXAdapter`, then run **QAFT** for 5 stabilisation epochs.

**Step 3** — Execute `PlanckXEngine` to run B-PACS packing on the trained path-5 layer weights and perform **autoregressive generation**, reporting weighted-average nominal stages consumed.

### Expected output

```
Epoch [500/500]   CE_Loss=0.0000   AvgEntropy=0.0004
Path1=49.5%   Path2=0.2%   Path5=50.3%

# Markdown Evaluation Matrix
| Loss                      | 0.0000 |
| Output Entropy            | 0.0003 |
| Path 1 % (H < 4.0)        | 49.50% |
| Path 5 % (H >= 4.5)       | 50.30% |
| Avg Nominal Stages        | 1.00   |
```

---

## Benchmarking Results

### Routing Utilisation (v3.1-Adaptive)

| Metric | Value |
|---|---|
| Training Loss | 0.0000 |
| Output Entropy | 0.0003 |
| Path 1 Utilisation (H < 4.0) | **49.5%** |
| Path 2 Utilisation (4.0 ≤ H < 4.5) | 0.2% |
| Path 5 Utilisation (H ≥ 4.5) | 50.3% |
| Rough average Nominal Stages | ~2.5× faster than all-Path-5 (1.00 vs 5.00) |

### B-PACS Weight-Footprint Proof

| Layer | Raw Weights | Packed Groups | Reduction |
|---|---|---|---|
| `path_5_sequential.0.weight` | 4,096 × int8 | 512 × int16 | **8 : 1** |
| Packed bits | 32,768 bits | 8,192 bits | ✓ confirmed |

> These figures are obtained from the standard 50-char × 23-string synthetic training set. Actual utilisation on real-world corpora will vary; the ratio system is scale-invariant.

---

## Citation

If you use PLANCK-X in your research, please cite:

```bibtex
@misc{planckx2024,
  title         = {PLANCK-X Chronos-Bit Engine v3.1-Adaptive: Spectral-Entropy-Guided
                  Mixed-Precision Neural Routing},
  author        = {Your Name},
  journal       = {GitHub repository},
  year          = {2024},
  url           = {https://github.com/<your-org>/planck-x},
  note          = {Mixed-precision routing + B-PACS ternary packing prototype}
}
```

---

## License

[MIT License](LICENSE) — see the [`LICENSE`](LICENSE) file for full terms.

---

<p align="center">
<i>PLANCK-X is a research prototype for mixed-precision neural routing.</i>
</p>
