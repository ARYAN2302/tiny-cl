# V11: SLAO + AVR Combination

Addresses the design tension between SLAO (which merges) and AVR (which repairs). V11 is the first version to combine them in a controlled experiment, with EWC baseline and reverse-order test.

## Methods tested

1. **naive**     — sequential fine-tuning, no CL
2. **ewc**       — EWC regularization (finally)
3. **slao**      — SLAO Algorithm 1 (ICLR 2026, arXiv 2512.23017)
4. **slao_avr**  — SLAO + AVR verify-repair after merge
5. **naive_ext** — naive + extra steps matching slao_avr's repair budget (compute-matched dummy)

## Design decisions

- **AVR repair target** = previous `merged_state` snapshot (constant memory), NOT the per-task fine-tuned state.
- **Repair fires AFTER SLAO merge** (clean causal story).
- **Track repair fire count** (zero is a real, reportable result).
- **PPL-ratio gate** (NOT hidden-state MSE). Fires if `ppl_now / ppl_best > 1.15`.
- **Closed-form repair** via weight interpolation: `θ ← (1−α)·θ + α·θ_snapshot` with α=0.1.

## Checks

- Multi-seed (42, 123, 456)
- Forward (A→B→C) and reverse (C→B→A) order
- Plasticity cost (newest domain PPL)
- Repair fire count per seed
- Module verification
- Compute-matched dummy baseline

## Files

- `v11_combo.py` — single-file Kaggle cell

## Status

Shipped. The AVR mechanism (PPL-ratio gate + closed-form repair) is reused in v23 standalone on TRACE.
