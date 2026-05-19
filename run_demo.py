"""
PLANCK-X Chronos-Bit Engine — v3.1-Adaptive — Top-level Demo Runner
=====================================================================
Three-step execution pipeline:
  Step 1 — Train a fresh PlanckXLayer from scratch (PlanckXTrainer, 500 ep,
           lr = 0.02, with Sparsity / Certainty Penalty to force low entropy).
  Step 2 — Quantise a dense reference model with PlanckXAdapter (90/10 ternary)
           and run QAFT fine-tuning.
  Step 3 — Run PlanckXEngine autoregressive generation and print the full
           evaluation matrix (Loss, Entropy, Path %, Avg Nominal Stages, Nuance & Consistency Score).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from planckx.routing import PlanckXLayer
from planckx.trainer import PlanckXTrainer
from planckx.adapter import PlanckXAdapter
from planckx.engine import PlanckXEngine

# ==================================================================== #
#  CONSTANTS                                                            #
# ==================================================================== #

VOCAB_SIZE: int = 23
D_MODEL: int = 64
SEQ_LEN: int = 8

# Training setup — aggressive for fast convergence into low-entropy zone
LEARNING_RATE: float = 0.02
EPOCHS_SCRATCH: int = 500
EPOCHS_QAFT: int = 5
GRAD_CLIP: float = 1.0
MAX_TOKENS_GEN: int = 20

# 23-character vocabulary  (idx -> char, char -> idx)
CHAR_SET: list = [
    "a", "b", "c",
    "d", "e", "f",
    "g", "h", "i",
    "j", "k", "l",
    "m", "n", "o",
    "p", "q", "r",
    "s", "t", "u",
    "v", "x",
]
assert len(CHAR_SET) == VOCAB_SIZE, (
    f"CHAR_SET length ({len(CHAR_SET)}) != VOCAB_SIZE ({VOCAB_SIZE})"
)
CHAR_TO_IDX: dict = {ch: i for i, ch in enumerate(CHAR_SET)}
IDX_TO_CHAR: dict = {v: k for k, v in CHAR_TO_IDX.items()}

# ── Dataset — one entry per vocab character, long strings ───────────────── #
#  Each string contains a single repeated character  (e.g. 50 'h's) so the
#  per-step CE loss is near-zero within a few epochs, but with 23 strings ×
#  ~43 sliding windows each the dataset has ~1 000 samples — enough total
#  evidence for the optimizer to actually push entropy below 4.0 and fire
#  Path 1 for at least some tokens.
RAW_STRINGS: list = [
    "a" * 50,  "b" * 50,  "c" * 50,  "d" * 50,  "e" * 50,
    "f" * 50,  "g" * 50,  "h" * 50,  "i" * 50,  "j" * 50,
    "k" * 50,  "l" * 50,  "m" * 50,  "n" * 50,  "o" * 50,
    "p" * 50,  "q" * 50,  "r" * 50,  "s" * 50,  "t" * 50,
    "u" * 50,  "v" * 50,  "x" * 50,
]

# ==================================================================== #
#  DATASET BUILDER                                                      #
# ==================================================================== #

def build_dataset() -> torch.Tensor:
    """
    Build a tokenised sliding-window dataset from RAW_STRINGS.

    Each raw string is expanded into every SEQ_LEN-length window, then
    all windows are concatenated into a single [N, SEQ_LEN] tensor.
    """
    token_seqs: list[list[int]] = []
    for raw in RAW_STRINGS:
        indices = [CHAR_TO_IDX[ch] for ch in raw]
        token_seqs.append(indices)

    window_size = SEQ_LEN
    windows: list[list[int]] = []
    for seq in token_seqs:
        for i in range(0, len(seq) - window_size + 1):
            windows.append(seq[i : i + window_size])

    n_windows = len(windows)
    dataset_np = np.zeros((n_windows, window_size), dtype=np.int64)
    for r, win in enumerate(windows):
        dataset_np[r] = win

    return torch.tensor(dataset_np, dtype=torch.long)


# ==================================================================== #
#  MODEL WRAPPER  (TokenEmbedding + PlanckXLayer + VocabProjection)     #
# ==================================================================== #

class PlanckXCharModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.plankx_layer = PlanckXLayer(d_model=d_model, vocab_size=vocab_size)
        self.output_projection = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(input_ids)        # [B, S-1, D]
        flat_embed = embedded.reshape(-1, self.d_model)   # [N, D]

        # Single-pass routing: use the router's lightweight routing_head to
        # compute routing entropy and dispatch tokens. This avoids a
        # preliminary full Path-5 execution and ensures unused branches
        # never execute.
        routed_flat = self.plankx_layer(
            flat_embed.reshape(embedded.shape),           # [B, S-1, D]
            entropy_logits=None,
        )
        logits = self.output_projection(routed_flat)
        return logits

    def reset_counters(self) -> None:
        self.plankx_layer.reset_counters()

    def get_utilization_rates(self) -> dict:
        return self.plankx_layer.get_utilization_rates()


# ==================================================================== #
#  DENSE REFERENCE MODEL (for adapter benchmark)                        #
# ==================================================================== #

class DenseCharModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.sequential_core = nn.Sequential(
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
        self.output_projection = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(input_ids)
        activated = self.sequential_core(embedded)
        logits = self.output_projection(activated)
        return logits


# ==================================================================== #
#  STEP 1 — TRAIN FROM SCRATCH                                          #
# ==================================================================== #

def step_1_train_from_scratch(dataset: torch.Tensor) -> tuple[PlanckXCharModel, list]:
    print("=" * 64)
    print("STEP 1 — PlanckXTrainer   |  epochs = 500   |  lr = 0.02")
    print("=" * 64)

    model = PlanckXCharModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    trainer = PlanckXTrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        dataset=dataset,
    )
    telemetry = trainer.train_scratch(epochs=EPOCHS_SCRATCH)

    final_loss = telemetry[-1]["loss"]
    final_entropy = telemetry[-1]["avg_entropy"]
    util = model.plankx_layer.get_utilization_rates()

    print("\n--- FINAL PERFORMANCE MATRIX ---")
    print(f"  Final Loss              :  {final_loss:.4f}")
    print(f"  Final Avg Entropy       :  {final_entropy:.4f}")
    print(f"  Path 1 Utilization      :  {util['path_1_pct']:.2f} %")
    print(f"  Path 2 Utilization      :  {util['path_2_pct']:.2f} %")
    print(f"  Path 5 Utilization      :  {util['path_5_pct']:.2f} %")

    return model, telemetry


# ==================================================================== #
#  STEP 2 — ASYMMETRIC QUANTISATION + QAFT                             #
# ==================================================================== #

def step_2_quantise_and_finetune(dataset: torch.Tensor) -> tuple[DenseCharModel, list]:
    print("\n" + "=" * 64)
    print("STEP 2 — PlanckXAdapter   |  90/10 ternary swap  +  QAFT (5 ep)")
    print("=" * 64)

    dense_model = DenseCharModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL)

    adapter = PlanckXAdapter(model=dense_model)
    adapter.apply_asymmetric_precision()

    q_summary = adapter.get_quantization_summary()
    print(
        f"  Total Linear Layers  : {q_summary['total_linear_layers']}"
        f"  (quantized: {q_summary['quantized_layers']},"
        f"  fp32 gating: {q_summary['float_layers']})"
    )

    qaft_log = adapter.fine_tune_adapter(dataset=dataset, epochs=EPOCHS_QAFT)
    return dense_model, qaft_log, q_summary


# ==================================================================== #
#  STEP 3 — INFERENCE + EVALUATION MATRIX                              #
# ==================================================================== #

@torch.no_grad()
def compute_output_entropy(model: nn.Module, dataset: torch.Tensor) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inputs = dataset[:, :-1]
    targets = dataset[:, 1:]
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    log_probs = torch.log2(torch.softmax(flat_logits, dim=-1) + 1e-9)
    entropy_per_token = -torch.sum(torch.softmax(flat_logits, dim=-1) * log_probs, dim=-1)
    return float(entropy_per_token.mean().item())


@torch.no_grad()
def compute_perplexity(model: nn.Module, dataset: torch.Tensor) -> float:
    """
    Compute perplexity of the model on the dataset.
    Perplexity = exp(average cross-entropy loss)
    Lower perplexity indicates better performance (less "nonsense").
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    inputs = dataset[:, :-1]
    targets = dataset[:, 1:]
    logits = model(inputs)
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    
    # Cross entropy loss
    loss = criterion(flat_logits, flat_targets)
    # Perplexity = exp(loss)
    perplexity = torch.exp(loss)
    return float(perplexity.item())


@torch.no_grad()
def compute_nuance_consistency_score(model: PlanckXCharModel, dataset: torch.Tensor) -> float:
    """
    Compute a 'Nuance & Consistency Score' by testing on a complex string.
    The score measures how well the model distinguishes between:
    - Obvious repetition (should use Path 1 - 1-stage bit-shift)
    - Complex/nuanced language (should use Path 5 - full precision)
    
    Returns a score between 0 and 1, where higher is better.
    """
    model.eval()
    
    # Create a test complex string with varied punctuation and case
    # Using characters from our vocabulary but arranging them in a complex pattern
    complex_chars = ['h', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']
    # Filter to only include chars in our vocabulary
    vocab_chars = set(CHAR_SET)
    filtered_chars = [c for c in complex_chars if c in vocab_chars]
    
    # If we don't have enough chars, create a pattern from available ones
    if len(filtered_chars) < 4:
        # Use first few available chars to create a repeating pattern
        base_chars = list(vocab_chars)[:4] if len(vocab_chars) >= 4 else list(vocab_chars)
        # Create a complex pattern: abcdefgh... but with variations
        pattern = ""
        for i in range(20):  # Make it 20 chars long
            pattern += base_chars[i % len(base_chars)]
        # Add some punctuation-like variation by changing case (though we don't have case in our vocab)
        # Instead, we'll insert spaces and other available chars periodically
        complex_string = ""
        for i, char in enumerate(pattern):
            complex_string += char
            # Periodically insert a different available char for variety
            if (i + 1) % 5 == 0 and len(vocab_chars) > 1:
                # Insert a different char from vocab
                other_chars = [c for c in vocab_chars if c != char]
                if other_chars:
                    complex_string += other_chars[0]
    else:
        # Use the filtered chars to create our complex string
        complex_string = "".join(filtered_chars) * 3  # Repeat to make it longer
    
    # Convert string to tensor indices
    try:
        complex_indices = [CHAR_TO_IDX[ch] for ch in complex_string if ch in CHAR_TO_IDX]
        if not complex_indices:
            # Fallback: use a simple repetitive string
            complex_indices = [CHAR_TO_IDX['a']] * 10
        complex_tensor = torch.tensor(complex_indices, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    except KeyError:
        # Fallback if any char not in vocab
        complex_tensor = torch.tensor([CHAR_TO_IDX['a']] * 10, dtype=torch.long).unsqueeze(0)
    
    # We need to compare against a simple repetitive string for obvious repetition
    simple_char = CHAR_SET[0]  # 'a'
    simple_string = simple_char * max(10, len(complex_string))
    simple_indices = [CHAR_TO_IDX[simple_char]] * len(simple_string)
    simple_tensor = torch.tensor(simple_indices, dtype=torch.long).unsqueeze(0)  # [1, seq_len]
    
    # Reset counters before testing
    model.reset_counters()
    
    # Test on complex string
    with torch.no_grad():
        # Forward pass to get routing information
        _ = model(complex_tensor)
    
    complex_util = model.get_utilization_rates()
    complex_p1 = complex_util["path_1_pct"]
    complex_p5 = complex_util["path_5_pct"]
    
    # Reset counters again
    model.reset_counters()
    
    # Test on simple repetitive string
    with torch.no_grad():
        _ = model(simple_tensor)
    
    simple_util = model.get_utilization_rates()
    simple_p1 = simple_util["path_1_pct"]
    simple_p5 = simple_util["path_5_pct"]
    
    # Nuance & Consistency Score:
    # For complex string: we want HIGH Path 5 usage (nuanced language handled with full precision)
    # For simple string: we want HIGH Path 1 usage (obvious repetition handled with bit-shift)
    # Score = (complex_path_5_pct + simple_path_1_pct) / 200  (normalized to 0-1)
    # But we need to handle case where percentages might be 0
    score = (complex_p5 + simple_p1) / 200.0
    
    # Ensure score is between 0 and 1
    score = max(0.0, min(1.0, score))
    
    return score


def step_3_generate_and_evaluate(
    trained_model: PlanckXCharModel,
    dataset: torch.Tensor,
) -> dict:
    print("\n" + "=" * 64)
    print("STEP 3 — PlanckXEngine  |  B-PACS packing  +  autoregressive gen")
    print("=" * 64)

    # --- B-PACS demonstration ----------------------------------------
    print("\n  -- B-PACS 8:1 Packing Simulation --")
    engine = PlanckXEngine(model=trained_model, vocab_size=VOCAB_SIZE)
    p5_weights = trained_model.plankx_layer.path_5_sequential[0].weight.data.detach().cpu().flatten()

    packed_real = engine.simulate_bpacs_packing(p5_weights.to(dtype=torch.int8))
    print(f"  Raw path-5 layer-1 weight count : {p5_weights.numel()}  (d_model x d_model = {D_MODEL}x{D_MODEL})")
    print(f"  Packed int16 array length       : {len(packed_real)}  groups")

    demo_weights = torch.tensor(
        [1, 0, -1, 0, 1, 1, -1, 0, 1, 0, 0, 1, -1, 1, 0, -1],
        dtype=torch.int8,
    )
    print(f"\n  -- Demo Bit-Pattern (first {min(16, demo_weights.numel())} ternary values) --")
    packed_demo = engine.simulate_bpacs_packing(demo_weights)
    for idx in range(len(packed_demo)):
        print(f"    [{idx:>2d}] = 0x{packed_demo[idx] & 0xFFFF:04X}")

    # --- Autoregressive generation -----------------------------------
    print(f"\n  -- Autoregressive Generation  |  max_tokens = {MAX_TOKENS_GEN} --")
    start_seq = dataset[0:1, :4]
    print(f"  Seed tokens : {[IDX_TO_CHAR[t.item()] for t in start_seq[0]]}")

    generated_ids, gen_meta = engine.generate(
        start_tokens=start_seq,
        max_tokens=MAX_TOKENS_GEN,
    )
    generated_chars = [IDX_TO_CHAR[t] for t in generated_ids]
    print(f"  Generated   : {generated_chars}")
    print(f"  Encoded IDs : {generated_ids}")

    # --- Counts for weighted avg nominal stages ------------------------
    step_stages = gen_meta["step_stages"]          # list of 1.0 / 2.0 / 5.0 per step
    count_p1 = sum(1 for s in step_stages if s == 1.0)
    count_p2 = sum(1 for s in step_stages if s == 2.0)
    count_p5 = sum(1 for s in step_stages if s == 5.0)
    total_generated = len(step_stages)
    # Avg_Nominal_Stages = (Count_P1 * 1 + Count_P2 * 2 + Count_P5 * 5) / Total
    avg_nominal_stages = (
        (count_p1 * 1.0 + count_p2 * 2.0 + count_p5 * 5.0) / max(total_generated, 1)
    )

    print(f"\n  [Gen counts]  P1={count_p1}  P2={count_p2}  P5={count_p5}"
          f"  avg_Nominal_Stages = {avg_nominal_stages:.2f}")

    # --- Evaluation matrix -------------------------------------------
    final_loss = 0.0
    criterion_eval = nn.CrossEntropyLoss()
    inputs_eval = dataset[:, :-1]
    targets_eval = dataset[:, 1:]
    logits_eval = trained_model(inputs_eval)
    final_loss = float(criterion_eval(
        logits_eval.reshape(-1, logits_eval.shape[-1]),
        targets_eval.reshape(-1),
    ).item())
    output_entropy = compute_output_entropy(trained_model, dataset)
    perplexity = compute_perplexity(trained_model, dataset)
    nuance_consistency_score = compute_nuance_consistency_score(trained_model, dataset)
    util = trained_model.plankx_layer.get_utilization_rates()

    matrix = {
        "loss": final_loss,
        "output_entropy": output_entropy,
        "perplexity": perplexity,
        "nuance_consistency_score": nuance_consistency_score,
        "path_1_pct": util["path_1_pct"],
        "path_2_pct": util["path_2_pct"],
        "path_5_pct": util["path_5_pct"],
        "avg_nominal_stages": avg_nominal_stages,
    }

    return matrix


# ==================================================================== #
#  MARKDOWN EVALUATION MATRIX                                          #
# ==================================================================== #

def print_markdown_matrix(matrix: dict) -> None:
    lines = [
        "## PLANCK-X Chronos-Bit Engine  |  Consolidated Evaluation Matrix",
        "",
        "| Metric                    | Value                  |",
        "|---------------------------|------------------------|",
        f"| Loss                      | {matrix['loss']:.4f}           |",
        f"| Output Entropy            | {matrix['output_entropy']:.4f}           |",
        f"| Perplexity                | {matrix['perplexity']:.4f}           |",
        f"| Nuance & Consistency Score| {matrix['nuance_consistency_score']:.4f}           |",
        f"| Path 1 % (H < 4.0)        |  {matrix['path_1_pct']:.2f} %         |",
        f"| Path 2 % (4.0 <= H < 4.5) |  {matrix['path_2_pct']:.2f} %         |",
        f"| Path 5 % (H >= 4.5)       |  {matrix['path_5_pct']:.2f} %         |",
        f"| Avg Nominal Stages        | {matrix['avg_nominal_stages']:.2f}             |",
        "",
    ]
    print("\n" + "\n".join(lines))


# ==================================================================== #
#  MAIN                                                                 #
# ==================================================================== #

def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    print()
    print("  PLANCK-X CHRONOS-BIT ENGINE  v3.1-Adaptive")
    print("  ========================================================")

    # --- Build synthetic dataset -------------------------------------
    dataset = build_dataset()
    print(f"\n  Dataset shape          : {tuple(dataset.shape)}")
    print(f"  Vocab size             : {VOCAB_SIZE}")
    print(f"  d_model (hidden dim)   : {D_MODEL}")
    print(f"  Sequence length        : {SEQ_LEN}")
    print(f"  Characters             : {CHAR_SET}")

    # Step 1 ──────────────────────────────────────────────────────────
    trained_model, scratch_telemetry = step_1_train_from_scratch(dataset)

    # Step 2 ──────────────────────────────────────────────────────────
    dense_model, qaft_log, q_summary = step_2_quantise_and_finetune(dataset)

    # Step 3 ──────────────────────────────────────────────────────────
    eval_matrix = step_3_generate_and_evaluate(trained_model, dataset)

    # ── Final Markdown evaluation table ──────────────────────────────
    print_markdown_matrix(eval_matrix)
    print("  Demo complete.")


if __name__ == "__main__":
    main()
