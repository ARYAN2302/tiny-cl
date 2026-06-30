# Architecture

How the three mechanisms — **SLAO**, **MVA**, and **AVR** — compose into a single *living* model on top of LFM2.5-350M.

## Base model

LiquidAI/LFM2.5-350M is a hybrid architecture:

- **6 attention layers** at positions `[2, 5, 8, 10, 12, 14]`
- **10 conv layers** at positions `[0, 1, 3, 4, 6, 7, 9, 11, 13, 15]`
- 350M parameters, vocab 65536, native transformers support (`>= 5.0.0`)

All adaptation is done via **LoRA** (`peft` library) attached to `in_proj` and `out_proj` only (26 modules total, rank 32). The base weights are never touched. This makes the entire living-model state expressible as a small set of LoRA A/B matrices that can be saved, merged, and rolled back cheaply.

## The three mechanisms

### 1. SLAO — continual learning

SLAO (Single LoRA Adaptive Orthogonal, Qiao & Mahdavi, ICLR 2026) prevents catastrophic forgetting by keeping the LoRA-A subspace orthonormal across tasks. Per task:

```
After training task t:
  1. Extract A_t (current LoRA-A weights)
  2. QR-decompose A_t  →  Q (orthonormal), R (upper-triangular)
  3. Replace A ← Q (re-orthonormalize)
  4. Initialize next task's B from current merged B
  5. Merge B: B_merged ← B_old + (1/√t) · (B_new − B_old)
```

The A-replacement keeps the input subspace stable (so old task gradients stay in span), while the B-interpolation averages task contributions with decaying weight.

**Known structural limit:** After T tasks, the first task's B contributes only `∏_{t=2}^{T} (1 − 1/√t)` of its trained magnitude — about **12.4%** at T=3. This is the floor SLAO cannot break without switching to product-space merging.

### 2. MVA — self-improvement

MVA (Model-Validated Adaptation) is our certainty-gated self-improvement loop. Per round:

```
1. Generate N answers on a fresh SQuAD sample (pass@5 sampling, T=0.7)
2. Compute certainty for each answer:
   certainty = KL(U ‖ p)            # INTUITOR-style
   where U = uniform,  p = model's answer distribution
3. Filter: keep only answers with certainty > τ
   τ = 50th percentile of THIS round's certainty distribution (adaptive)
4. Train on the filtered set (3 epochs, LoRA)
5. SLAO-merge the resulting weights into the living model
```

**The honest framing:** the certainty gate is *not* a correctness gate. It filters out 60–88% of wrong answers (precision 79–94%), which reduces wrong-answer contamination from ~30% (naive) to ~6–21% (MVA). The pass@5 improvement reflects training on fewer mistakes, not on verified-correct self-generated data.

### 3. AVR — drift detection and repair

AVR (Anchor-Verify-Repair) runs alongside SLAO and provides a *post-hoc* safety net. Per task:

```
After training task t:
  1. Snapshot the LoRA state S_t
  2. Compute best PPL on each prior task j: ppl_best[j] = min(ppl_best[j], ppl_after_t)
  3. Verify: for each prior task j, check ppl_now[j] / ppl_best[j]
     If ratio > 1.15 → task j has drifted
  4. Repair (closed-form, no optimizer):
     θ ← (1 − α) · θ + α · θ_snapshot        # α = 0.1
     Repeat until no task drifts, or MAX_REPAIR_STEPS reached
  5. Continue to next task
```

The repair step is a **convex combination** in weight space. It pulls the model toward the snapshot state for drifted tasks, without requiring gradient computation or task labels. The interpolation factor α=0.1 was tuned in v11.

## How they compose

```
            ┌─────────────────────────────────────┐
            │   LFM2.5-350M  (frozen base)         │
            └─────────────┬───────────────────────┘
                          │
                  ┌───────▼────────┐
                  │  LoRA adapters  │ (rank=32, in_proj+out_proj)
                  └───────┬────────┘
                          │
        ┌─────────────────┼─────────────────────┐
        │                 │                     │
   ┌────▼───┐        ┌────▼────┐         ┌──────▼─────┐
   │  SLAO  │        │   MVA   │         │    AVR     │
   │ (merge)│        │ (gate)  │         │ (verify+   │
   │        │        │         │         │  repair)   │
   └────┬───┘        └────┬────┘         └──────┬─────┘
        │                 │                     │
        └─────────────────┼─────────────────────┘
                          │
                  ┌───────▼────────┐
                  │  Living model   │
                  │  state (LoRA)   │
                  └────────────────┘
```

- **SLAO** runs after every training round (domain or MVA) and merges LoRA-B weights
- **MVA** runs as a 4th, 5th, 6th round on top of the SLAO-merged model (v14/v15)
- **AVR** runs as a verification step after each task and can rewind the model toward a snapshot (v23)

## Data flow

| Source                  | Used by        | Size                |
|-------------------------|----------------|---------------------|
| `epfl-llm/guidelines`   | v13 (domain A) | 1M tokens, medical  |
| `iamtarun/python_code_instructions_18k_alpaca` | v13 (domain B) | 1M tokens, code |
| `roneneldan/TinyStories`| v13 (domain C) | 1M tokens, creative |
| SQuAD                   | v14, v15 (MVA) | 200 questions/round |
| TRACE benchmark (4 tasks) | v18, v23     | 5000-example variant |

All datasets are downloaded automatically by the scripts (HuggingFace Hub + gdown for TRACE).

## Reproducibility

- **Seed:** 42 for v13/v14/v15/v18/v23; 123 for `self_improvement_validation.py`
- **Determinism:** `torch.manual_seed`, `numpy.random.seed`, `random.seed` all set; CUDA determinism not enforced (acceptable — variance reported across seeds)
- **Precision:** bfloat16 throughout (no FP32 fallback)
- **Hardware target:** single T4 GPU (Kaggle), 16GB VRAM
- **Wall-clock:** 1.5–5 hours per experiment depending on configuration
