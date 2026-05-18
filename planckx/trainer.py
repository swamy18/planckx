import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanckXTrainer:
    """
    PlanckXTrainer

    Orchestrates end-to-end training with an explicit Sparsity / Certainty Penalty
    that is backpropagated through the EXACT same logits the CE loss is computed
    from, ensuring the entropy gate driving PlanckXLayer routing is directly
    coupled to the optimisation objective.

    Loss :  total = CE_loss + ENTROPY_COEFF * mean(H(X))
    Opt  :  Adam, grad-clip max_norm = 1.0.
    """

    ENTROPY_COEFF: float = 0.2    # strong enough to visibly push H down

    def __init__(self, model: nn.Module, learning_rate: float, dataset: torch.Tensor):
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if not isinstance(dataset, torch.Tensor):
            raise TypeError("dataset must be a torch.Tensor")

        self.model = model
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.dataset_size = dataset.shape[0]

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        self.telemetry_log: list = []

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _compute_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Shannon Entropy H = -sum(p_i * log2(p_i + 1e-9)); shape [N]."""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log2(probs + 1e-9)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def _collect_telemetry(
        self, epoch: int, loss_val: float, avg_entropy: float, utilization: dict
    ) -> None:
        self.telemetry_log.append({
            "epoch": epoch,
            "loss": loss_val,
            "avg_entropy": avg_entropy,
            "path_1_pct": utilization.get("path_1_pct", 0.0),
            "path_2_pct": utilization.get("path_2_pct", 0.0),
            "path_5_pct": utilization.get("path_5_pct", 0.0),
        })

    # ================================================================== #
    #  Main training loop                                                  #
    # ================================================================== #

    def train_scratch(self, epochs: int = 300) -> list:
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        self.model.train()
        self.model.reset_counters()

        inputs = self.dataset[:, :-1]       # [N, SEQ_LEN-1]
        targets_all = self.dataset[:, 1:]   # [N, SEQ_LEN-1]

        for epoch in range(1, epochs + 1):
            self.optimizer.zero_grad()

            # --------------- Forward pass -------------------------------
            # logits are produced by PlanckXCharModel which itself runs
            # a preliminary forward pass to obtain routing entropy from
            # the SAME output_projection weights the CE loss is computed on.
            logits = self.model(inputs)                     # [N, S-1, vocab_size]
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets_all.reshape(-1)

            # --------------- CE Loss -----------------------------------
            ce_loss = self.criterion(flat_logits, flat_targets)

            # --------------- Entropy Penalty (BACKPROPAGATED) -----------
            # H is computed directly from those same flat_logits so the
            # gradient from ENTROPY_COEFF * H flows back through
            # output_projection and every intermediate layer, including
            # the router inside PlanckXLayer.
            entropy_per_token = self._compute_entropy_from_logits(flat_logits)
            mean_entropy = entropy_per_token.mean()
            sparsity_penalty = self.ENTROPY_COEFF * mean_entropy

            # --------------- Combined total + backward -----------------
            total_loss = ce_loss + sparsity_penalty
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # --------------- Telemetry (no-grad) -----------------------
            with torch.no_grad():
                probs = F.softmax(flat_logits, dim=-1)
                log_p = torch.log2(probs + 1e-9)
                token_entropy = -(probs * log_p).sum(dim=-1)
                current_loss = ce_loss.item()
                avg_entropy = token_entropy.mean().item()

            utilization = self.model.get_utilization_rates()

            if (epoch % 50) == 0:
                self._collect_telemetry(epoch, current_loss, avg_entropy, utilization)
                print(
                    f"  Epoch [{epoch:>4d}/{epochs}]"
                    f"  CE_Loss={current_loss:.4f}"
                    f"  AvgEntropy={avg_entropy:.4f}"
                    f"  Path1={utilization['path_1_pct']:.1f}%"
                    f"  Path2={utilization['path_2_pct']:.1f}%"
                    f"  Path5={utilization['path_5_pct']:.1f}%"
                )

        print(
            f"\n  Training complete — Final CE_Loss: "
            f"{self.telemetry_log[-1]['loss']:.4f}"
        )
        return self.telemetry_log
