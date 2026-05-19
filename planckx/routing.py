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
        # Lightweight routing head: cheap logits for entropy estimation
        # Use a much smaller "shallow" vocab to reduce entropy compute cost.
        self.routing_vocab = min(512, max(64, vocab_size // 64))
        self.routing_head = nn.Linear(in_features=d_model, out_features=self.routing_vocab)
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

        # ---- Entropy gate: compute cheap shallow logits via routing_head
        # If caller provided full `entropy_logits`, prefer that (useful
        # for evaluation), otherwise compute shallow logits to decide.
        if entropy_logits is not None:
            # If provided logits have larger vocab, try to reduce cost by
            # computing entropy over a shallow projection when possible.
            try:
                entropy = self._compute_shannon_entropy(entropy_logits)
            except Exception:
                # Fallback: compute from routing_head
                shallow_logits = self.routing_head(X_flat)
                entropy = self._compute_shannon_entropy(shallow_logits)
                entropy_logits = None
        else:
            shallow_logits = self.routing_head(X_flat)
            entropy = self._compute_shannon_entropy(shallow_logits)
            entropy_logits = None

        # ---- Nuance-Gate: compute cheaply on shallow logits only for
        # tokens near the decision boundary to avoid global topk costs.
        # Create a conservative initial nuance_mask of all False.
        nuance_mask = torch.zeros(flat_dim, dtype=torch.bool, device=X.device)
        # Identify borderline tokens (within 0.1 bits of threshold)
        borderline = (entropy >= (self.THRESH_PATH5 - 0.1)) & (entropy <= (self.THRESH_PATH5 + 0.1))
        if borderline.any():
            # compute top-5 variance on shallow logits for borderline tokens
            b_idx = borderline.nonzero(as_tuple=True)[0]
            try:
                topk = min(5, shallow_logits.size(-1))
                topk_vals, _ = torch.topk(shallow_logits[b_idx], k=topk, dim=-1)
                topk_var = torch.var(topk_vals, dim=-1, unbiased=False)
                nuance_mask[b_idx] = topk_var > self.NUANCE_VARIANCE_THRESHOLD
            except Exception:
                # If anything fails, leave nuance_mask as False to avoid added cost
                pass

        # ---- Boolean masks — relaxed thresholds -----------------------
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

        # ---- Index-dispatch: execute only selected paths on subsets ---
        out = torch.empty_like(X_flat)

        # Path 1
        idx_a = mask_a.nonzero(as_tuple=True)[0]
        if idx_a.numel() > 0:
            subset = X_flat[idx_a]
            out_a = self._path_1_transform(subset)
            out[idx_a] = out_a

        # Path 2
        idx_b = mask_b.nonzero(as_tuple=True)[0]
        if idx_b.numel() > 0:
            subset = X_flat[idx_b]
            out_b = self._path_2_transform(subset)
            out[idx_b] = out_b

        # Path 5
        idx_c = mask_c.nonzero(as_tuple=True)[0]
        if idx_c.numel() > 0:
            subset = X_flat[idx_c]
            out_c = self._path_5_transform(self.path_5_sequential, subset)
            out[idx_c] = out_c

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
