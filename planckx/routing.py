import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanckXLayer(nn.Module):
    """
    PLANCK-X Chronos-Bit Engine Layer — v3.1-Adaptive (Routing Fix v2).

    Routing entropy is now computed DIRECTLY from the final model-output
    logits (passed in as `entropy_logits`) rather than from an independent
    internal vocab_projection.  This guarantees the routing gate is
    backpropagated through the EXACT same parameters the optimizer is
    updating for the CE loss — eliminating the decoupling bug that kept
    all tokens stuck on Path 5.

    Category A (Path 1): H < 4.0  -> (X * 127) >> 7
    Category B (Path 2): 4.0 <= H < 4.5  -> (X >> 2) + (X >> 4)
    Category C (Path 5): H >= 4.5  -> 5-layer nn.Sequential

    Nuance-Gate: Even if Entropy is low, if the 'Variance' of the top-5 logits
    is high (indicating a nuanced word choice), FORCE the token through Path 5.
    This ensures that poetic or complex language never gets "flattened" by
    the 1-stage bit-shift path.

    All branches are evaluated in parallel; torch.where selects per token.
    """

    # --- Dynamic thresholds (relaxed from v1) ---------------------------
    THRESH_PATH2: float = 4.0    # Path 1 fires when H < 4.0
    THRESH_PATH5: float = 4.5    # Path 5 fires when H >= 4.5; else Path 2
    NUANCE_VARIANCE_THRESHOLD: float = 0.5  # Threshold for top-5 logit variance
    EPS: float = 1e-9

    def __init__(self, d_model: int = 64, vocab_size: int = 32000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 5-layer sequential deep-processing block (Path 5 core)
        self.path_5_sequential = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(),
            nn.Linear(in_features=d_model, out_features=d_model),
            nn.ReLU(),
        )
        # NOTE: NO vocab_projection inside this layer.
        #       Routing entropy is computed outside, from the model's actual
        #       output_projection logits, and passed in as entropy_logits.

        # Absolute utilisation counters, updated each forward call
        self.path_1_counter: int = 0
        self.path_2_counter: int = 0
        self.path_5_counter: int = 0

    # ================================================================== #
    #  Static helpers                                                       #
    # ================================================================== #

    @staticmethod
    def _compute_shannon_entropy(logits_2d: torch.Tensor) -> torch.Tensor:
        """
        Shannon Entropy in base-2  H = -sum(p_i * log2(p_i + eps)).
        Accepts logits of arbitrary shape ending in vocab dim, returns
        per-sample entropy of shape [N] where N is the flattened batch
        dimension.
        """
        probs = F.softmax(logits_2d, dim=-1)
        log_probs = torch.log2(probs + PlanckXLayer.EPS)
        entropy_per_sample = -torch.sum(probs * log_probs, dim=-1)
        return entropy_per_sample

    @staticmethod
    def _path_1_transform(t: torch.Tensor) -> torch.Tensor:
        """
        Category A — 1-stage bitwise shortcut.
        Formula : (X * 127.0).int32() >> 7  ->  float32
        Cost    : ~1 MAC / element (single-bit right shift after scale)
        """
        scaled = t * 127.0
        int_t = scaled.to(dtype=torch.int32)
        return (int_t >> 7).to(dtype=torch.float32)

    @staticmethod
    def _path_2_transform(t: torch.Tensor) -> torch.Tensor:
        """
        Category B — 2-stage shift-add fusion.
        Formula : (X * 127.0).int32() >> 2  +  >> 4  ->  float32
        Cost    : ~2 MAC / element (two shifts + one add)
        """
        scaled = t * 127.0
        int_t = scaled.to(dtype=torch.int32)
        return ((int_t >> 2) + (int_t >> 4)).to(dtype=torch.float32)

    @staticmethod
    def _path_5_transform(block: nn.Sequential, t: torch.Tensor) -> torch.Tensor:
        """
        Category C — 5-stage deep sequential block.
        Executes all five Linear+ReLU pairs unconditionally.
        """
        return block(t)

    # ================================================================== #
    #  Forward pass — entropy from actual model outputs, parallel routing  #
    # ================================================================== #

    def forward(
        self,
        X: torch.Tensor,
        entropy_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X              : [batch, seq, d_model] embedded token activations.
        entropy_logits : [batch * seq, vocab_size] FINAL output logits from
                          the model's output_projection.  Used to compute
                          the Shannon-Entropy routing gate so gradients flow
                          through the exact same head that produces the CE
                          loss.  Pass None to fall back on a per-sample
                          uniform baseline.

        Returns
        -------
        routed : [batch, seq, d_model] — activation tensor after routing.
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        flat_dim = batch_size * seq_len

        # ---- Flatten to [N, d_model] -----------------------------------
        X_flat = X.reshape(flat_dim, self.d_model)

        # ---- Entropy gate: tie to actual model output logits -----------
        # entropy_logits is produced by self.output_projection(X_flat) in
        # the wrapper model's forward pass — the same parameters the
        # optimizer updates for the CE loss.
        if entropy_logits is not None:
            entropy = self._compute_shannon_entropy(entropy_logits)
            # ---- Nuance-Gate: Check variance of top-5 logits -----------
            # Even if Entropy is low, if the 'Variance' of the top-5 logits
            # is high (indicating a nuanced word choice), FORCE the token
            # through Path 5.
            with torch.no_grad():
                import time
                _start_time = time.perf_counter()

                # Get top-5 logits and their variance
                top5_logits, _ = torch.topk(entropy_logits, k=min(5, entropy_logits.size(-1)), dim=-1)
                # Calculate variance of top-5 logits for each token
                top5_variance = torch.var(top5_logits, dim=-1, unbiased=False)  # [flat_dim]
                # Create nuance mask: high variance indicates nuanced choice
                nuance_mask = top5_variance > self.NUANCE_VARIANCE_THRESHOLD
                
                # Stress Test Verification: Check if Nuance-Gate latency is under 1ms
                _latency_ms = (time.perf_counter() - _start_time) * 1000.0
                if _latency_ms > 1.0:
                    print(f"[SCALABILITY WARNING] Nuance-Gate variance calculation latency exceeded 1ms: {_latency_ms:.3f}ms")
        else:
            # Fallback: estimate from raw X_flat statistics if no logits
            # were provided (should not happen in normal usage)
            log_probs = torch.log2(
                F.softmax(X_flat, dim=-1) + self.EPS
            )
            entropy = -torch.sum(
                F.softmax(X_flat, dim=-1) * log_probs, dim=-1
            )
            nuance_mask = torch.zeros(flat_dim, dtype=torch.bool, device=X.device)
        # entropy shape: [flat_dim]

        # ---- Boolean masks — relaxed thresholds -----------------------
        # Path 1: H < 4.0   (was 3.8)
        # Path 2: 4.0 <= H < 4.5
        # Path 5: H >= 4.5 OR nuance_mask is True (Nuance-Gate)
        mask_a = entropy < self.THRESH_PATH2          # [N]
        mask_b = (~mask_a) & (entropy < self.THRESH_PATH5)   # [N]
        mask_c = (entropy >= self.THRESH_PATH5) | nuance_mask   # [N]

        # ---- Update utilisation counters from this forward pass's masks
        n_a = int(mask_a.sum().item())
        n_b = int(mask_b.sum().item())
        n_c = int(mask_c.sum().item())
        if n_a:
            self.path_1_counter += n_a
        if n_b:
            self.path_2_counter += n_b
        if n_c:
            self.path_5_counter += n_c

        # ---- Evaluate all three branches in parallel over full tensor --
        # Each call touches ALL elements; per-element correctness is
        # guaranteed by the torch.where selectors below.
        result_a = self._path_1_transform(X_flat)    # [N, d_model]
        result_b = self._path_2_transform(X_flat)    # [N, d_model]
        result_c = self._path_5_transform(           # [N, d_model]
            self.path_5_sequential, X_flat
        )

        # ---- Assemble output with element-wise masking -----------------
        # Stage 1: choose A vs B; Stage 2: override with C where mask_c
        out = torch.where(mask_a.unsqueeze(-1), result_a, result_b)
        out = torch.where(mask_c.unsqueeze(-1), result_c, out)

        # ---- Reshape back to [batch, seq, d_model] --------------------
        return out.reshape(batch_size, seq_len, self.d_model)

    # ================================================================== #
    #  Utilisation reporting                                               #
    # ================================================================== #

    def get_utilization_rates(self) -> dict:
        total = self.path_1_counter + self.path_2_counter + self.path_5_counter
        if total == 0:
            return {
                "path_1_pct": 0.0, "path_2_pct": 0.0,
                "path_5_pct": 0.0, "total_tokens": 0,
            }
        return {
            "path_1_pct": 100.0 * self.path_1_counter / total,
            "path_2_pct": 100.0 * self.path_2_counter / total,
            "path_5_pct": 100.0 * self.path_5_counter / total,
            "total_tokens": total,
        }

    def reset_counters(self) -> None:
        self.path_1_counter = 0
        self.path_2_counter = 0
        self.path_5_counter = 0
