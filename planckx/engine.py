import torch
import torch.nn as nn
import numpy as np


_TER_TO_INT = np.array(
    [-1, 0, 1],
    dtype=np.int16,
)

_BPACS_SHIFT = np.array([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int16)

_BPACS_MASK = np.array(0x03, dtype=np.uint16)


class PlanckXEngine:
    """
    PlanckXEngine

    Standalone edge-inference engine that:
      1. Simulates B-PACS 8:1 ternary-weight packing into 16-bit containers.
      2. Runs autoregressive token generation with hardware-path routing
         driven by live entropy measurement at every step.
      3. Tracks the running Average Nominal Stages consumed across the
         generated sequence.
    """

    def __init__(self, model: nn.Module, vocab_size: int = 23):
        self.model = model
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------ #
    #  B-PACS PACKING                                                     #
    # ------------------------------------------------------------------ #

    def simulate_bpacs_packing(self, weights: torch.Tensor) -> np.ndarray:
        if weights.dim() != 1:
            raise ValueError("weights must be a 1-D tensor")
        n = weights.shape[0]
        if n % 8 != 0:
            pad = 8 - (n % 8)
            weights = torch.cat([weights, torch.zeros(pad, dtype=weights.dtype)])
        w_np = weights.cpu().detach().numpy().astype(np.int8)
        n_groups = w_np.shape[0] // 8
        packed = np.zeros(n_groups, dtype=np.int16)

        for g in range(n_groups):
            group = w_np[g * 8 : (g + 1) * 8]
            packed_g = np.int16(0)
            for slot in range(8):
                raw_val = int(group[slot])
                if raw_val == 1:
                    code = np.int16(1)
                elif raw_val == -1:
                    code = np.int16(2)
                else:
                    code = np.int16(0)
                shift_amount = int(_BPACS_SHIFT[slot])
                packed_g = packed_g | (code << shift_amount)
            packed[g] = packed_g

        print(f"  [B-PACS] Input length: {n} ternary values")
        print(f"  [B-PACS] Groups of 8: {n_groups}  => packed into {n_groups} x int16")
        print(f"  [B-PACS] Packing ratio: 8:1  |  raw bits: {n * 8}  packed bits: {n_groups * 16}")
        return packed

    # ------------------------------------------------------------------ #
    #  GENERATION                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_tokens: int = 20,
    ) -> tuple[list, dict]:
        self.model.eval()
        self.model.reset_counters()

        if start_tokens.dim() == 1:
            context = start_tokens.unsqueeze(0)
        else:
            context = start_tokens.clone()

        generated_token_ids: list = []
        stage_accumulator = []

        for step in range(max_tokens):
            seq_len_now = context.shape[1]
            model_out = self.model(context[:, -1:])

            step_logits = model_out[:, -1, :]
            probs = torch.softmax(step_logits, dim=-1)
            log_probs = torch.log2(probs + 1e-9)
            entropy_step = float((-torch.sum(probs * log_probs, dim=-1)).mean().item())

            if entropy_step < 2.5:
                nominal_stage = 1.0
            elif entropy_step < 3.5:
                nominal_stage = 2.0
            else:
                nominal_stage = 5.0

            stage_accumulator.append(nominal_stage)

            next_token = torch.argmax(step_logits, dim=-1)
            next_token_scalar = int(next_token.item())
            generated_token_ids.append(next_token_scalar)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)

        cum_stages = sum(stage_accumulator)
        avg_nominal = cum_stages / max(len(stage_accumulator), 1)

        meta = {
            "avg_nominal_stages": avg_nominal,
            "step_entropies": None,
            "step_stages": stage_accumulator,
        }
        return generated_token_ids, meta
