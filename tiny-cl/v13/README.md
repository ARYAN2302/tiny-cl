# V13 — The Full Picture

This is the final continual-learning experiment. Five methods, five domain orderings, multi-seed, with cross-term noise measurement. Result: **SLAO r=32 ships** with FF(A) = 1.097 ± 0.007.

## Files

| File | What |
|------|------|
| `v13_full_picture.py` | One-cell Kaggle script. Self-installs deps. ~4-5h on T4. |
| `v13_kaggle_output.txt` | Raw Kaggle output — every method × seed × ordering. |

## Methods tested

1. **naive** — sequential SFT, no continual learning. Floor.
2. **slao** — SLAO Algorithm 1 (Qiao & Mahdavi, ICLR 2026). **Shipped.**
3. **fixed_a** — freeze A after task 1, train B only. Diagnostic for cross-term noise.
4. **dw_stitch** — merge in ΔW space via QR + 2r×2r SVD. Memory-efficient. **Has a bug** (additive B_cat instead of interpolative). Kept as negative result.
5. **slao_r64** — SLAO at rank 64. Capacity check.

## Results (3 seeds, forward A→B→C)

| Method     | FF(A) μ±σ         | FF(B) μ±σ         | New PPL μ±σ       | Noise% |
|------------|-------------------|-------------------|-------------------|--------|
| naive      | 1.517 ± 0.020     | 1.406 ± 0.022     | 6.003 ± 0.115     | 195.6% |
| **slao**   | **1.097 ± 0.007** | **1.065 ± 0.005** | 6.348 ± 0.112     | 191.1% |
| fixed_a    | 1.149 ± 0.005     | 1.107 ± 0.006     | 6.437 ± 0.107     | 196.1% |
| dw_stitch  | 207.768 ± 54.980  | 26.442 ± 12.543   | 9300.7 ± 3548.5   | 190.2% |
| slao_r64   | 1.091 ± 0.015     | 1.057 ± 0.009     | 6.231 ± 0.090     | 190.2% |

## Domain ordering (seed=42, SLAO)

| Order    | FF(first) | FF(second) |
|----------|-----------|------------|
| A→B→C    | 1.09      | 1.06       |
| C→B→A    | 1.19      | 1.03       |
| B→C→A    | 1.15      | 1.10       |
| A→C→B    | 1.03      | 1.05       |
| C→A→B    | 1.19      | 0.97       |

Medical (A) is the most fragile. Creative-first is the worst curriculum.

## Why these numbers

- **Cross-term noise is NOT the bottleneck.** Fixed-A eliminates it mathematically and is still worse than SLAO (1.149 vs 1.097). The interleaved A re-orthonormalization in SLAO does real work.
- **Capacity is NOT the bottleneck.** Doubling rank from 32 to 64 barely moves the needle (1.097 → 1.091).
- **The bottleneck is B interpolation dilution.** After 3 tasks, the first task's B carries only `(1-1/√2)·(1-1/√3) ≈ 12.4%` of its trained weight. This is the structural limit of SLAO.
- **ΔW-stitch in product space is the mathematically correct fix** but the current implementation uses additive `B_cat = [B_old, λ·B_new]` which double-counts energy. The fix is interpolative `B_cat = [(1-λ)·B_old, λ·B_new]`. Not re-tested — user decision: ship SLAO, move on.

## Decision

**SLAO r=32 is the shipped method.** FF(A) = 1.097× is "good enough" for the living-model claim. The 5% gap to the 1.05× target is understood and structural. We do not pursue further CL methods.

Next phase: **self-improvement without forgetting**, building on top of SLAO — see `../v14/` (SLAO + 1 MVA round), `../v15/` (SLAO + 3 MVA rounds), `../v18/` (TRACE benchmark), and `../v23/` (AVR standalone on TRACE). Design notes in [`../../NEXT_STEPS.md`](../../NEXT_STEPS.md).
