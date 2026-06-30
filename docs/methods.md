# Methods

Formal definitions, equations, and hyperparameters for the three mechanisms.

---

## SLAO — Single LoRA Adaptive Orthogonal

**Reference:** Qiao & Mahdavi, ICLR 2026 (arXiv 2512.23017).

**Goal:** Continual learning with a single LoRA adapter, no per-task expansion, no replay buffer.

### Setup

Each task `t` produces a fine-tuned LoRA pair `(A_t, B_t)`, where `A_t ∈ ℝ^{r×d_in}` and `B_t ∈ ℝ^{d_out×r}`. After training, the adapter is merged into a single living state `(A, B)`.

### Algorithm (per task)

```
1. Train task t with LoRA initialized from (A, B) = (A_{t-1}, B_{t-1})
2. After training, extract A_t
3. QR-decompose: A_t^T = Q R,  with Q ∈ ℝ^{d_in×r} orthonormal
   Fix signs: Q ← Q · diag(sign(diag(R)))
4. Replace: A_t ← Q^T
5. Merge B:
   B_merged = B_{t-1} + λ_t · (B_t − B_{t-1})
   where λ_t = 1/√t
6. Set living state: (A, B) ← (A_t, B_merged)
```

### Why it works

- **A-replacement** keeps the input subspace orthonormal across tasks, so old-task gradients stay in span(A). This is the anti-forgetting step.
- **B-interpolation** averages new-task knowledge into the existing B with decaying weight `λ_t = 1/√t`. After T tasks, the first task's contribution to B is `∏_{t=2}^{T} (1 − 1/√t)`.

### Hyperparameters (v13/v14/v15/v18/v23)

| Parameter       | Value                          |
|-----------------|--------------------------------|
| LoRA rank `r`   | 32                             |
| LoRA alpha      | 32                             |
| LoRA dropout    | 0.05                           |
| Target modules  | `["in_proj", "out_proj"]`      |
| Optimizer       | AdamW, lr=2e-4, wd=0.01        |
| Max grad norm   | 1.0                            |
| Epochs per task | 1 (v13), 3 (v18/v23)           |
| Precision       | bfloat16                       |

### Known structural limit

After 3 tasks (T=3), the first task's B contributes `(1 − 1/√2)·(1 − 1/√3) ≈ 12.4%` of its trained magnitude. This is the source of the residual 10% forgetting observed in v13. Doubling rank (r=64) does not help — capacity is not the bottleneck.

### What does *not* work (negative results from v13)

1. **Fixed-A** (freeze A after task 1, train B only): eliminates cross-term noise mathematically but still worse than SLAO (FF(A)=1.149 vs 1.097). The interleaved A re-orthonormalization does real work.
2. **ΔW-stitch in product space** (merge `ΔW = B·A` directly via QR + 2r×2r SVD): mathematically correct, but our additive `B_cat = [B_old, λ·B_new]` implementation double-counts energy. The fix is interpolative `B_cat = [(1−λ)·B_old, λ·B_new]`. Not re-tested.

---

## MVA — Model-Validated Adaptation

**Inspiration:** INTUITOR (Zhao et al., ICLR 2026) — self-certainty as an intrinsic reward.

**Goal:** Self-improvement on a held-out task (SQuAD) without external reward model or ground-truth labels.

### Setup

Given a model `M` and a set of prompts `P` (SQuAD questions), generate `k=5` answers per prompt via temperature sampling, then filter and train.

### Algorithm (per round)

```
1. For each prompt p ∈ P:
   a. Generate k=5 answers via T=0.7 sampling
   b. For each answer y_i, compute certainty:
      certainty(y_i) = KL(U ‖ p_M(·|p, y_i))      # INTUITOR self-certainty
      where U is uniform over the answer token positions
   c. Record (p, y_i, certainty_i)

2. Compute threshold τ = percentile_50({certainty_i})    # adaptive per round
3. Filter: keep only (p, y_i) where certainty_i > τ
4. Train M on the filtered set (3 epochs, LoRA)
5. SLAO-merge the resulting weights (Algorithm 1 above)
```

### The honesty clause

The certainty gate is **not** a correctness gate. Measured precision is 79–94% across seeds, meaning 6–21% of training examples are still wrong. The mechanism that produces the pass@5 gain is **reduced-error filtering**, not **correct-data generation**. We assert only the former.

### Hyperparameters (v14/v15)

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Prompts per round      | 200 (SQuAD, fresh sample per round)|
| Samples per prompt     | 5 (pass@5)                         |
| Sampling temperature   | 0.7                                |
| Certainty signal       | `KL(U ‖ p)` (INTUITOR-style)       |
| Threshold              | 50th percentile of round's distribution (adaptive) |
| Training epochs        | 3                                  |
| Validation: consensus  | not used in v14/v15 (was in v5)    |

### Why adaptive threshold

The fixed threshold `τ=17.0` (tuned on base model in v5) produced **0/200** validated pairs on the post-SLAO model. SLAO shifts the certainty distribution: base mean=16.68 → post-SLAO mean=11.58. The adaptive (50th-percentile) threshold makes the filter operational across distribution shifts.

---

## AVR — Anchor-Verify-Repair

**Origin:** v11 (PPL-ratio gate, NOT hidden-state MSE).

**Goal:** Post-hoc detection and repair of catastrophic forgetting, without replay data and without gradient computation.

### Setup

Maintain a per-task "best PPL" table `ppl_best[j]` and a snapshot of the LoRA state `S_t` after each completed task.

### Algorithm (per task)

```
1. Snapshot the current LoRA state S_t
2. For each completed task j, compute current PPL on task j's eval set
3. Update ppl_best[j] = min(ppl_best[j], ppl_after_t)
4. VERIFY:
   for each prior task j:
     ratio = ppl_now[j] / ppl_best[j]
     if ratio > 1.15:
       mark j as drifted
5. REPAIR (while any task drifted and steps < MAX_REPAIR_STEPS):
   θ ← (1 − α) · θ + α · θ_snapshot       # α = 0.1
   recompute ppl_now[j] for all j
   unmark tasks whose ratio is now ≤ 1.15
6. Proceed to next task
```

### Why it works

- **Verify** uses PPL ratio as a model-free drift detector. No hidden-state statistics, no Fisher information, no replay. The 1.15 threshold corresponds to ~14% log-perplexity increase, which is empirically the smallest drift that signals meaningful forgetting on 350M.
- **Repair** is a **closed-form weight interpolation** — no optimizer, no gradients, no labels. It pulls the model toward the snapshot state, which is guaranteed to have lower PPL on the drifted task (by construction of `ppl_best`).
- The repair step is **idempotent under no further training**: once `θ = θ_snapshot`, repair has no effect.

### Hyperparameters (v23)

| Parameter              | Value      |
|------------------------|------------|
| Drift threshold        | 1.15 (PPL ratio) |
| Repair α               | 0.1        |
| Max repair steps       | 100 (practically never hit) |
| Snapshot granularity   | Per-task (LoRA state at end of each task) |

### What AVR does *not* do

- **No task-id inference at inference time.** AVR only acts during training, not at test time.
- **No replay buffer.** The snapshot is just a LoRA state dict (a few hundred MB at most).
- **No gradient computation during repair.** The repair is closed-form.
- **No protection against forward transfer degradation.** AVR preserves PPL on prior tasks but does not optimize for new-task performance.

---

## Composing the three

| Experiment | SLAO | MVA | AVR | Benchmark  |
|------------|------|-----|-----|------------|
| v13        | ✓    | —   | —   | 3 domains  |
| v14        | ✓    | 1 round | — | 3 domains + SQuAD |
| v15        | ✓    | 3 rounds | — | 3 domains + SQuAD |
| v18        | ✓    | ✓   | —   | TRACE (4 tasks) |
| v23        | —    | —   | ✓   | TRACE (4 tasks) |
| self_improvement_validation | — | ✓ | — | SQuAD (standalone) |

Note that v23 runs AVR *standalone* (no SLAO, no MVA). This isolates AVR's contribution. A combined SLAO+AVR configuration is planned but not yet run.

---

## Evaluation metrics

### Forgetting Factor (FF) — used in v13/v14/v15

For a task `t`, FF measures how much perplexity has increased relative to its best:

```
FF(t) = ppl_now(t) / ppl_best(t)
```

- `FF = 1.0`: no forgetting
- `FF = 1.5`: 50% perplexity increase (severe forgetting)
- `FF > 2.0`: catastrophic

### ACC / BWT / FWT — used in v18/v23 (TRACE)

Standard metrics from the GEM benchmark (NeurIPS 2017):

Let `R[i,j]` = accuracy on task `j` after training task `i`.

- **ACC** = mean of last row of `R` (overall performance after all training)
- **BWT** (backward transfer) = `mean(R[T,j] − R[j,j])` for `j < T` (how much old tasks degraded)
  - Negative = forgetting; zero = no forgetting; positive = improvement
- **FWT** (forward transfer) = `mean(R[i−1,i] − baseline[i])` for `i > 0` (how much new tasks improved)

### pass@5 — used in self_improvement_validation

Standard pass@k: probability that at least one of `k` samples is correct.

```
pass@k = 1 − C(n−c, k) / C(n, k)
```

where `n` is the total number of samples and `c` is the number of correct samples.
