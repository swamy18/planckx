# PLANCK-X Conditional Execution Audit

Date: 2026-05-19

Summary
-------
This audit locates where the current PLANCK-X implementation executes all compute paths in parallel, documents the primary causes of wasted FLOPs and latency, and recommends immediate fixes to enable true conditional execution (execute only the selected path).

Key findings
------------
- Primary root cause: the router performs two full passes (a conservative full-path pass to obtain logits/entropy, then a second routed pass). See [run_demo.py](run_demo.py#L110-L210). This forces the expensive Path 5 to run for every token at least once.

- Core router evaluates all branches unconditionally and assembles outputs with `torch.where`, causing all path computations to happen for every token. See [planckx/routing.py](planckx/routing.py#L1-L235). Examples:
  - `result_a = self._path_1_transform(X_flat)`
  - `result_b = self._path_2_transform(X_flat)`
  - `result_c = self._path_5_transform(self.path_5_sequential, X_flat)`
  - `out = torch.where(mask_a.unsqueeze(-1), result_a, result_b); out = torch.where(mask_c.unsqueeze(-1), result_c, out)`

- `torch.where` masking is used for element-wise selection but does not prevent the evaluation of unused branches — it only selects after all branches produced tensors.

- Entropy/logit extraction is currently produced by running the full-path `PlanckXLayer` (or by materializing full logits from `output_projection` after a full pass). This doubles the work. See [run_demo.py](run_demo.py#L110-L210).

- Nuance-Gate computes `topk` and `var` with `torch.topk` and `torch.var` (in `torch.no_grad()`), which are expensive operations on large vocabularies and may introduce hidden synchronization or large memory spikes. See [planckx/routing.py](planckx/routing.py#L1-L235).

- vLLM adapter and other integration points call the router expecting it to be "masked-lightweight"; but because branches are computed unconditionally, integration still incurs full compute. See [planckx/vllm_adapter.py](planckx/vllm_adapter.py#L1-L74).

- Extra allocations: `X_flat`, intermediate `result_*` tensors, integer casts in `_path_1_transform/_path_2_transform` (creating int tensors), and intermediate softmax/prob tensors when computing entropy all increase peak VRAM.

- Training loop and telemetry call the model twice per forward (to obtain routing entropy and then to compute final routed output), doubling FLOPs and latency. See [planckx/trainer.py](planckx/trainer.py#L41-L110) and [run_demo.py](run_demo.py#L110-L210).

Severity and impact
-------------------
- The two-pass pattern (full pass for entropy + routed pass) and unconditional branch evaluation are the dominant sources of wasted compute. Measured symptom (reported earlier): routed inference ~4x slower on CPU vs baseline — consistent with this doubling (and more) of work.

Concrete evidence locations
--------------------------
- Router parallel branch evaluation: [planckx/routing.py](planckx/routing.py#L1-L235)
- Two-pass conservative routing in the model wrapper: [run_demo.py](run_demo.py#L110-L210)
- vLLM integration assuming masked-lightweight behavior: [planckx/vllm_adapter.py](planckx/vllm_adapter.py#L1-L74)
- Training & telemetry that rely on the double-forward pattern: [planckx/trainer.py](planckx/trainer.py#L41-L110)
- Project documentation describing tri-path behavior: [README.md](README.md#L41-L120)

Immediate recommendations (Phase 2 readiness)
--------------------------------------------
1. Eliminate the preliminary full-path pass used solely to compute entropy.
   - Option A (preferred): Add a lightweight routing head that computes `entropy_logits` cheaply from the *pre-routing activations* (a single Linear layer or a very small MLP). This head must be significantly cheaper than Path 5 and be used only to compute routing decisions.
   - Option B (short-term): Cache logits from previous time-step(s) or use a single-step projection `output_projection_shallow(X_flat)` that is much cheaper than running `path_5`.

2. Replace unconditional branch evaluation with index-based conditional execution:
   - Compute `mask_a/mask_b/mask_c` (boolean vector) first.
   - Use `torch.nonzero(mask).squeeze(-1)` (or `mask.nonzero(as_tuple=True)[0]`) to get token indices for each path.
   - For each path, `if indices.numel() > 0: subset = X_flat[indices]; result_subset = path_fn(subset); write_back via out[indices] = result_subset`.
   - This pattern guarantees unused branches never execute and avoids creating full `result_*` tensors.

3. Avoid `torch.where` for routing assembly; perform indexed scatter/gather or direct assignment into an output buffer to avoid materializing full branch outputs.

4. Move Nuance-Gate computations off the critical path or make them cheaper:
   - Consider computing top-k on a reduced candidate set (e.g., narrowed vocabulary subset), or compute a cheaper proxy (e.g., top-1 vs top-2 gap) instead of full top-5 variance.
   - If variance check is required, perform it only for tokens near threshold (i.e., coarse entropy binning first), not for all tokens.

5. Minimize temporary allocations and dtype conversions:
   - Implement bit-shift transforms using integer views only when necessary; prefer integer ops in-place when safe.
   - Avoid repeated `to(dtype=...)` on CUDA without `non_blocking` considerations; prefer performing datatype-limited ops on CPU only if that avoids GPU stalls.

6. Add explicit profiling hooks and microbenchmarks around the following hotspots:
   - Entropy computation (softmax + log + sum)
   - Nuance-Gate topk + var
   - Path 5 execution (per-token time)
   - Index selection and scatter/gather overhead
   - Any CUDA<->CPU transfers

Suggested short-term patch (Phase 2 plan outline)
--------------------------------------------------
- Implement a `RoutingHead(nn.Module)` with a single `nn.Linear(d_model, vocab_shallow)` (e.g., vocab_shallow = 512 or 1024). Use this to compute a cheap `entropy_logits_shallow` and gating masks.
- In `PlanckXLayer.forward()`:
  1. Flatten X -> X_flat
  2. Compute `entropy_logits_shallow = routing_head(X_flat)` and `entropy = compute_entropy(entropy_logits_shallow)`
  3. Compute masks `mask_a/mask_b/mask_c`
  4. Allocate `out = torch.empty_like(X_flat)` once
  5. For each path where `indices.numel() > 0`: compute `out[indices] = path_fn(X_flat[indices])`
  6. Reshape `out` back to `[batch, seq, d_model]`
- Remove the preliminary conservative full pass in `run_demo.py`; obtain entropy from the routing head instead of a full `PlanckXLayer` pass.

Profiling & validation checklist (for Phase 5)
-----------------------------------------------
- Add timers and torch.cuda.synchronize() wrappers (when on GPU) around the following to collect accurate timings:
  - Routing head forward
  - Entropy computation
  - Per-path execution times and counts
  - Index gather/scatter overhead
  - Nuance-Gate cost per token
- Track peak VRAM before/after each forward and the peak during path 5 execution.
- Add FLOP approximator (estimated MACs) per path and compute realized MACs for routed runs.

Next steps
----------
- Implement Phase 2 changes (RoutingHead + index-dispatch + remove conservative pass) and run microbenchmarks comparing:
  A. Baseline dense model
  B. Old parallel masked router
  C. New index-dispatch conditional router

- I can start implementing Phase 2 now (modify `planckx/routing.py` and `run_demo.py` and add a `routing_head`), then run a micro-benchmark on CPU. Proceed?

Appendix — files inspected
-------------------------
- [planckx/routing.py](planckx/routing.py#L1-L235)
- [run_demo.py](run_demo.py#L110-L210)
- [planckx/vllm_adapter.py](planckx/vllm_adapter.py#L1-L74)
- [planckx/trainer.py](planckx/trainer.py#L41-L110)
- [README.md](README.md#L41-L120)
