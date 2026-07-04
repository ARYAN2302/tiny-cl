# avr-cl

**The first continual post-training framework for LLMs.**

Llama-Factory, Unsloth, Axolotl, TRL — they all do single-stage post-training. None of them handle the sequential case. Train SFT, then DPO, then add a new domain, and they quietly destroy what came before.

avr-cl wraps the full post-training pipeline as a stream. LEARN → VERIFY → REPAIR, each phase pluggable. AVR v1 is the default instance. v2 research (subspace repair, DPO/GRPO, oracles) slots in without API changes.

---

## The design

```
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  LEARN  │────▶│ VERIFY  │────▶│ REPAIR  │
    │ (train) │     │ (drift) │     │ (rewind)│
    └─────────┘     └─────────┘     └─────────┘
         │                                 │
         └─────────── ◀───────────────────┘
                       (if drifted)
```

Three phases, each pluggable:

| Phase | v1 (shipped) | v2 (research) |
|---|---|---|
| LEARN | SFT (LoRA) | DPO, GRPO |
| VERIFY | PPL-ratio (threshold 1.15) | KL, Hessian, entropy |
| REPAIR | snapshot interpolation (α=0.1) | subspace repair (probe-gradient SVD) |

Plus two future-facing stubs (noop today, real implementations in v2):
- **Oracle** — external verifier for grounding (ExecOracle, SympyOracle)
- **Consolidator** — promote working state to long-term (VerifyGatedConsolidator)

The stubs exist in the API so v2 work lands without breaking anything.

---

## AVR v1: the default instance

**Anchor-Verify-Repair.** After training each task:

1. **Snapshot** the LoRA state.
2. **Verify**: for each prior task, compute PPL on its eval set. If `PPL_now / PPL_best > 1.15`, the task has drifted.
3. **Repair**: `θ ← (1-α)·θ + α·θ_snapshot` — closed-form interpolation toward the snapshot. No gradients, no replay, no labels at repair time.

**Adaptive improvements over the v23 paper config:**
- α decays with stream position: `α_eff = α / √(task_index + 1)`. Reduces over-repair on later tasks.
- max_steps scales with drift magnitude: `ceil(log(ratio) / log(1/(1-α)))`. Stops the 65% reversion problem when drift is small.

---

## Quickstart

```bash
pip install avr-cl
avr train configs/trace_lfm350m.yaml
```

Or programmatically:

```python
from avr.strategy import AVRStrategy
from avr.data import load_trace
from avr.cli import create_model, load_config

config = load_config("configs/trace_lfm350m.yaml")
model, tokenizer = create_model(config)
tasks = load_trace("results/")

strategy = AVRStrategy(config)
state = strategy.run(model, tokenizer, tasks)
print(f"BWT improvement: {state.total_repair_steps} repair steps applied")
```

---

## Results (TRACE, seed 42, LFM2.5-350M + LoRA r=32)

| Method | ACC | BWT | FF | Repair steps | Memory | Gradients at repair |
|---|---|---|---|---|---|---|
| Naive SFT | 0.379 | −0.130 | 0.130 | — | O(1) | — |
| Replay (10%) | _pending_ | _pending_ | _pending_ | — | O(N) | — |
| SLAO+MVA (ICLR 2026) | 0.397 | −0.062 | 0.062 | — | O(1) | — |
| **AVR (ours)** | **0.374** | **−0.023** | **0.038** | 24 | O(1) | **No** |

- 5.6× less forgetting than naive, 2.7× less than SLAO+MVA (BWT).
- Constant memory, no replay buffer.
- Repair is 24 closed-form weight interpolations across 4 tasks — no optimizer, no backward pass.

**Pending:** multi-seed (42/123/7), LFM2.5-1.2B, Qwen2.5-0.5B, Llama-3.2-1B, MMLU stream, replay baseline, distillation comparison. See `configs/`.

---

## Configs

Every run is one YAML file:

```yaml
model:
  id: LiquidAI/LFM2.5-350M
  lora_targets: [in_proj, out_proj]
  lora_rank: 32

stream:
  benchmark: trace
  tasks: [C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds]
  seed: 42

learn:
  method: sft          # pluggable: sft, replay_sft, (v2: dpo, grpo)
  epochs: 3
  lr: 2e-4

verify:
  detector: ppl_ratio  # pluggable: ppl_ratio, (v2: kl, hessian)
  threshold: 1.15
  probe_samples: 200

repair:
  operator: snapshot_interp  # pluggable: snapshot_interp, (v2: subspace_snapshot_interp)
  alpha: 0.1
  alpha_decay: sqrt
  max_steps_mode: adaptive

oracle: noop          # v2: exec, sympy, router
consolidator: noop    # v2: verify_gated
```

Shipped configs:
- `trace_lfm350m.yaml` — AVR on TRACE, LFM2.5-350M (the v23 reproduction)
- `trace_lfm350m_multiseed.yaml` — same, for `--seed 42/123/7` runs
- `trace_lfm1b.yaml` — scale up within LFM family
- `trace_qwen05b.yaml` — cross-family (pure Transformer decoder)
- `trace_llama1b.yaml` — cross-family, "most tunable" per distillabs
- `trace_replay.yaml` — replay baseline (10% buffer, 200 samples/task)
- `mmlu_stream.yaml` — second benchmark (8 MMLU subjects, T=8)
- `full_pipeline.yaml` — SFT → domain stream under AVR (Phase 2 demo, placeholder)

---

## The v2 research path

The framework is built so v2 work slots in as new subclasses, not API changes.

**Subspace repair (the next big idea).** AVR v1 repairs globally — every parameter gets pulled back by the same α. v2 repairs only the load-bearing subspace:

```
Δθ = θ_new - θ_snapshot             # what SFT just learned
g_probe = ∇_θ PPL_probe             # which directions the probe cares about
Δθ_load = proj(Δθ → span(g_probe))  # component that hurts the probe
θ_repaired = θ_new - α · Δθ_load    # repair only the load-bearing part
```

Implementation: probe-gradient SVD per LoRA layer (52 tiny SVDs, fast on T4). The `SubspaceSnapshotInterp` class is stubbed in `avr/operators.py` with the full implementation plan in comments.

**Other v2 work:**
- DPO/GRPO as LEARN strategies (swap `method: dpo` in config)
- KL/Hessian detectors (swap `detector: kl` in config)
- ExecOracle for code tasks (swap `oracle: exec` in config)
- VerifyGatedConsolidator for promotion (swap `consolidator: verify_gated`)

---

## Repository

```
avr/
├── __init__.py          # public API
├── framework.py         # ContinualPostTrainer + interfaces + Oracle/Consolidator stubs
├── strategy.py          # AVRStrategy — builds trainer from config
├── operators.py         # SnapshotInterp (v1), SubspaceSnapshotInterp (v2 stub)
├── detectors.py         # PPLRatioDetector (v1), KL/Hessian (v2 stubs)
├── trainer.py           # SFTStrategy, ReplaySFTStrategy, (v2: DPO, GRPO)
├── metrics.py           # R-matrix, BWT, FF, ACC, evaluate_task_accuracy
├── data.py              # TRACE, MMLU stream loaders
├── cli.py               # `avr train config.yaml`
├── configs/             # YAML configs (8 shipped)
├── pyproject.toml       # pip install avr-cl
└── results/             # output JSONs
```

---

## Status

- **Phase 0 (package): DONE.** Framework ships with pluggable LEARN/VERIFY/REPAIR + Oracle/Consolidator stubs. All smoke tests pass.
- **Phase 1 (credibility): NEXT.** Multi-seed TRACE, cross-model (LFM/Qwen/Llama), MMLU stream. Kaggle T4.
- **Phase 2 (pipeline demo):** SFT → domain stream under AVR. The breakthrough artifact.
- **Phase 3 (scale):** One 7B run on Modal.
- **Phase 4 (ship):** PyPI, r/LocalLLaMA, HN.

---

## License

MIT. The AVR method is original to this project. SLAO+MVA baseline is from Qiao & Mahdavi (ICLR 2026) and INTUITOR (ICLR 2026). TRACE benchmark is from Wang et al.
