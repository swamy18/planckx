import torch
import torch.nn as nn


TER_POS = 1.0
TER_NEG = -1.0
TER_ZERO = 0.0


class PlanckXAdapter:
    """
    PlanckXAdapter

    Converts a standard dense-floating-point PyTorch model into an asymmetric
    ternary-quantised variant using a 90 / 10 split:

      - 90 % of the model's nn.Linear layers  -> strict 3-state ternary weights
        (+1 / -1 / 0) for the deep-path processing core.
      - 10 % of the layer weights              -> left in full fp32 precision
        to stabilise gating operations.

    Provides QAFT (Quantization-Aware Fine-Tuning) to recover accuracy
    after the hard weight assignment.
    """

    def __init__(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module instance")
        self.model = model
        self._linear_keyframes: list = []
        self._quantized_count = 0
        self._total_linears = 0
        self._float_linear_names: set = set()

    def apply_asymmetric_precision(self) -> None:
        lines = self._collect_linear_layers()

        if len(lines) < 2:
            return

        q_count = max(1, int(round(0.90 * len(lines))))

        print(
            f"  [Adapter] Quantizing {q_count}/{len(lines)} linear layers "
            f"(90 % ternary, remainder fp32)."
        )

        self._model_and_keyframe_store: dict = {}

        for i, (name, module) in enumerate(lines):
            should_quantize = i < q_count

            with torch.no_grad():
                weight_data = module.weight.data.clone()
                if should_quantize:
                    pos_mask = weight_data > 0.5
                    neg_mask = weight_data < -0.5
                    weight_data = torch.where(pos_mask, torch.tensor(TER_POS), weight_data)
                    weight_data = torch.where(neg_mask, torch.tensor(TER_NEG), weight_data)
                    weight_data[~(pos_mask | neg_mask)] = TER_ZERO
                    module.weight.data.copy_(weight_data)
                    self._quantized_count += 1

                    self._linear_keyframes.append(
                        (name, module.weight.data.clone(), True)
                    )
                    self._model_and_keyframe_store.setdefault(name, {})["quantized"] = True

                else:
                    self._float_linear_names.add(name)

                    self._linear_keyframes.append(
                        (name, module.weight.data.clone(), False)
                    )
                    self._model_and_keyframe_store.setdefault(name, {})["quantized"] = False

        self._total_linears = len(lines)

    def _collect_linear_layers(self) -> list:
        collected = []
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                collected.append((layer_name, layer))
        return collected

    def fine_tune_adapter(self, dataset: torch.Tensor, epochs: int = 5) -> list:
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        if not isinstance(dataset, torch.Tensor):
            raise TypeError("dataset must be a torch.Tensor")

        if not hasattr(self, "_quantized_count"):
            self.apply_asymmetric_precision()

        q_linear_count = 0
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                q_linear_count += 1

        if q_linear_count == 0:
            return []

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        log_history = []

        self.model.train()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            running_samples = 0

            for i in range(dataset.shape[0]):
                optimizer.zero_grad()

                sample = dataset[i : i + 1]
                inputs = sample[:, :-1]
                targets = sample[:, 1:]

                logits = self.model(inputs)
                loss = criterion(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=1.0
                )
                optimizer.step()

                running_loss += loss.item()
                running_samples += 1

            avg_loss = running_loss / max(running_samples, 1)
            log_history.append({"epoch": epoch, "avg_loss": avg_loss})
            print(
                f"  [QAFT] Epoch [{epoch:>2d}/{epochs}]    Avg_Loss={avg_loss:.6f}"
            )

        print("  [QAFT] Fine-tuning stabilization complete.")
        return log_history

    def get_quantization_summary(self) -> dict:
        return {
            "total_linear_layers": self._total_linears,
            "quantized_layers": self._quantized_count,
            "float_layers": self._total_linears - self._quantized_count,
            "float_linear_names": sorted(self._float_linear_names),
        }
