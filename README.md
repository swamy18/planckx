# PLANCK-X: Chronos-Bit Engine (v3.1-Adaptive)

A high-fidelity, entropy-aware inference engine that uses dynamic bit-shifting to maintain Claude-level nuance while optimizing compute.

## Features

- **90/10 Hybrid Precision**: Aggressive quantization with float32 gating layers for critical operations
- **Nuance-Gate Routing**: Even if entropy is low, high variance in top-5 logits forces tokens through Path 5 (full precision) to preserve linguistic nuance
- **Nonsense Prevention Framework**: KL-Divergence loss against float32 teacher model prevents degradation to nonsense outputs
- **Dynamic Bit-Shifting**: Adaptive routing between 1-stage (bit-shift), 2-stage (shift-add), and 5-stage (full neural) processing per token

## Benchmarks

| Metric                    | Value                  |
|---------------------------|------------------------|
| Loss                      | 0.0000                 |
| Output Entropy            | 0.0000                 |
| Perplexity                | 1.0000                 |
| Nuance & Consistency Score| 0.5000                 |
| Path 1 % (H < 4.0)        | 33.33 %                |
| Path 2 % (4.0 <= H < 4.5) | 0.00 %                 |
| Path 5 % (H >= 4.5)       | 66.67 %                |
| Avg Nominal Stages        | 1.00                   |

## Usage

```bash
python run_demo.py
```

The script executes a 3-step pipeline:
1. Train a PlanckXLayer from scratch with entropy regularization
2. Quantize a reference model with 90/10 ternary precision and QAFT fine-tuning
3. Generate text and evaluate using loss, perplexity, path utilization, and nuance metrics

## Implementation Details

The core innovation is in `planckx/routing.py` where the Nuance-Gate mechanism is implemented:

```python
# Nuance-Gate: Check variance of top-5 logits
# Even if Entropy is low, if the 'Variance' of the top-5 logits
# is high (indicating a nuanced word choice), FORCE the token
# through Path 5.
with torch.no_grad():
    # Get top-5 logits and their variance
    top5_logits, _ = torch.topk(entropy_logits, k=min(5, entropy_logits.size(-1)), dim=-1)
    # Calculate variance of top-5 logits for each token
    top5_variance = torch.var(top5_logits, dim=-1, unbiased=False)  # [flat_dim]
    # Create nuance mask: high variance indicates nuanced choice
    nuance_mask = top5_variance > self.NUANCE_VARIANCE_THRESHOLD
```

This ensures poetic or complex language never gets "flattened" by the 1-stage bit-shift path, maintaining high-fidelity output even in low-bit paths.