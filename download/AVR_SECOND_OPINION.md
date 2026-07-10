# AVR Status & Second Opinion Request

## The big goal

Build a continual post-training framework (`avr-cl`) that lets you fine-tune an LLM on a stream of tasks without forgetting. Ship as a pip-installable package. The method is AVR (Anchor-Verify-Repair) — detect drift after training, repair it, no replay buffer, constant memory, no gradients at repair time (in v1).

**The small goal right now:** make AVR work robustly across seeds on TRACE (LFM2.5-350M, LoRA r=32). v23 with seed 42 gave ACC=0.374, BWT=−0.023, FF=0.038 — good. But seed 123 collapsed. Fix the seed-robustness problem before shipping.

---

## The specific bar the user set

> "Up the acc while maintaining this ff and btw."

Reference point: AVR interpolation on seed 123 gave ACC=0.261, BWT=−0.002, FF=0.005. The ACC collapsed because repair ran 100-205 steps and reverted 99.99% to the snapshot. The user wants ACC back up to ~0.35+ **without** losing the BWT=−0.002 / FF=0.005 retention. Both axes. Not a tradeoff.

---

## The method (AVR)

```
For each task T in stream:
  1. LEARN: SFT on task T (LoRA, rank 32, lr 2e-4, 3 epochs)
  2. SNAPSHOT: save LoRA weights
  3. VERIFY: compute PPL on all prior tasks; if PPL_now / PPL_best > 1.15 → drift
  4. REPAIR: fix the drift (mechanism varies — see versions below)
  5. Repeat for next task
```

The repair mechanism is what's being iterated. Everything else (model, LoRA, TRACE, drift detection) is fixed.

---

## Versions tried and what happened

### v23 — interpolation repair (the validated baseline)

**Repair mechanism:** `θ ← (1-α)·θ + α·θ_snapshot` where α=0.1. Blind weight interpolation toward the snapshot. Re-verify PPL after each step, stop when drift resolves or max_steps (100) hit.

**Seed 42:**
| ACC | BWT | FF | Repairs |
|-----|-----|----|---------|
| 0.374 | −0.023 | 0.038 | 24 |

- Repair log: 0/4/10/10 steps. Converged before the cap.
- This is the shipped result. It works on mild drift.

**Seed 123:**
| ACC | BWT | FF | Repairs |
|-----|-----|----|---------|
| 0.261 | −0.002 | 0.005 | 205 |

- Repair log: 0/5/100/100 steps. Hit the cap on tasks 3 and 4.
- 100 steps × α=0.1 = 1 − 0.9^100 = 99.997% reversion to snapshot.
- Old tasks: perfect retention (BWT=−0.002). New tasks: destroyed (NumGLUE-cm=0.000, NumGLUE-ds=0.045).
- **Diagnosis: over-repair. The blind interpolation reverted everything, including the new task's learning.**

---

### v24 — gradient repair (replace interpolation with optimization)

**Repair mechanism:** gradient steps on `loss = old_probe_loss + λ·new_task_loss`. λ=0.5 (balanced). lr=1e-4. 15 steps. No convergence check, fixed step count.

**Hypothesis:** interpolation is blind (pulls all directions equally). Gradient repair optimizes the tradeoff — finds directions that recover old tasks with minimal new-task damage.

**Seed 123:**
| ACC | BWT | FF | Repairs |
|-----|-----|----|---------|
| 0.412 | −0.085 | 0.085 | 45 |

- ACC recovered (+0.151 vs interpolation s123). Good.
- BWT collapsed (−0.085 vs −0.002). Bad — 42× worse.
- **Per-task logging showed:** old_loss and new_loss both decline early, then old_loss plateaus or reverses (step 6→11 on task 4: old_loss 5.50 → 5.83). The two losses fight each other.
- **Diagnosis: under-repair.** 15 steps at lr=1e-4 isn't enough budget to recover from 3× drift. Old tasks stayed drifted. BWT=−0.085 reflects that.

---

### v24.1 — gradient repair + λ decay + per-task logging (diagnostic)

**Changes from v24:**
- λ decays 0.5 → 0.1 over the repair steps (old-task recovery dominates late)
- Per-task old loss logging (see which task is the outlier)
- Log every step instead of every 5

**Seed 123:**
| ACC | BWT | FF | Repairs |
|-----|-----|----|---------|
| 0.405 | −0.092 | 0.092 | 45 |

- λ decay eliminated the reversal — old_loss declines monotonically now.
- Per-task losses are uniform (no outlier task dragging the average).
- **But still under-repair.** All losses still declining at step 15 when repair stops. Post-repair PPLs: none of the 3 drifted tasks recovered below the 1.15× threshold.
- **Diagnosis: the mechanism works, just needs more budget.** Losses are declining uniformly, no tug-of-war reversal. 15 steps can't fix 3× drift.

---

### v25 — gradient repair + bidirectional stop (the "fix")

**Changes from v24.1:**
- max_steps: 15 → 50 (more budget)
- Bidirectional stop every 5 steps:
  - Convergence: if all old tasks drop below 1.15× threshold → stop
  - New-task safeguard: if new-task PPL rises above 1.15× of its post-training best → stop immediately

**Seed 123:**
| ACC | BWT | FF | Repairs |
|-----|-----|----|---------|
| 0.433 | −0.058 | 0.058 | 85 |

- Repair log: 0/15/20/50 steps.
- Task 2: converged at 15 (old tasks recovered).
- Task 3: new-task safeguard fired at 20 (PPL 2.36 > 2.00 × 1.15). Stopped before new task collapsed.
- Task 4: ran full 50 steps, never converged. Old tasks: C-STANCE 1.24×, FOMC 1.49×, NumGLUE 1.51× — none below 1.15×. New task stayed safe (1.03× the whole time).
- **ACC is the best we've seen (0.433). BWT is worse than v23-s42 (−0.058 vs −0.023) but better than v24 (−0.085).**
- **Did NOT meet the user's bar.** BWT went from −0.002 (interpolation s123) to −0.058. That's not "maintaining BWT." The bidirectional stop picked a different Pareto point, it didn't break the tradeoff.

---

## All results at a glance

| Version | Mechanism | Seed | ACC | BWT | FF | Repairs | Notes |
|---------|-----------|------|-----|-----|----|---------|-------|
| Naive | None | 42 | 0.379 | −0.130 | 0.130 | — | No CL baseline |
| SLAO+MVA | Merge-during-train | 42 | 0.397 | −0.062 | 0.062 | — | Published ICLR 2026 baseline |
| v23 interp | Blind θ-interp, α=0.1 | 42 | 0.374 | −0.023 | 0.038 | 24 | ✓ Works on mild drift |
| v23 interp | Blind θ-intp, α=0.1 | 123 | 0.261 | −0.002 | 0.005 | 205 | ✗ Over-repair, ACC collapse |
| v24 gradient | Grad on old+λ·new, 15 steps | 123 | 0.412 | −0.085 | 0.085 | 45 | ACC up, BWT collapsed |
| v24.1 gradient+λdecay | Same + λ 0.5→0.1, 15 steps | 123 | 0.405 | −0.092 | 0.092 | 45 | Under-repair, no reversal |
| v25 gradient+bidir | Same + 50 steps + bidir stop | 123 | 0.433 | −0.058 | 0.058 | 85 | Best ACC, still not maintaining BWT |

---

## What's been diagnosed

1. **Interpolation over-repairs on severe drift.** 100 steps × α=0.1 = 99.997% reversion. Kills the new task. Fix: cap steps or add safeguard. (Not yet tried: simple 10-step cap on interpolation.)

2. **Gradient repair under-repairs at 15 steps.** Losses still declining when repair stops. Old tasks never recover. Fix: more steps. (Tried in v25 — 50 steps, still didn't fully recover 3× drift.)

3. **The zero-sum wall is real at r=32.** Both interpolation and gradient repair hit it. Interpolation navigates it blindly (overshoots). Gradient repair navigates it with optimization (finds a different point). Neither breaks it. The user's goal — ACC up AND BWT maintained — requires breaking the wall, not navigating it.

4. **v2 subspace repair didn't work either (prior work).** At r=32 on 350M, the load-bearing subspace IS the whole 32-dim LoRA update space. No orthogonal room for SFT to live in. Subspace projection doesn't free up anything. (This was established before the current diagnostic loop.)

5. **The new-task safeguard works.** v25 task 3: safeguard fired at step 20, stopped repair before new task collapsed. NumGLUE-cm stayed at 0.383 (not 0.000). The mechanism is sound — it just doesn't solve the zero-sum problem, it prevents the worst outcome.

---

## What has NOT been tried

1. **Interpolation with a 10-step cap.** The simplest fix for v23's over-repair. 10 steps × α=0.1 = 65% reversion. Old tasks get partial recovery, new task keeps 35%. Might hit ACC~0.35, BWT~−0.02. Not tried because the user said "10 will fix it but it's not an actual fix" — but it might be the shippable band-aid.

2. **Higher LR for gradient repair.** v25 used lr=1e-4. Losses declined but didn't converge in 50 steps. 5e-4 might converge faster. Risk: overshoot. Not tried because the user's second opinion said "hold off on the 5× LR bump until you've separated per-task old_loss" — and we did that, and the per-task losses were uniform, but we never went back to try the LR bump.

3. **Hybrid: interpolate first, then gradient.** Use interpolation for the first 5 steps (fast, cheap, recovers the easy drift), then switch to gradient for the remaining steps (targeted, recovers the hard drift). Not tried.

4. **Per-task repair.** Instead of one repair loop for all drifted tasks, do separate repair loops per drifted task. Might reduce the tug-of-war when 3 old tasks are competing. Not tried.

5. **Correction adapter (additive capacity).** Add a separate small r=8 adapter dedicated to old-task correction, leave the main LoRA untouched. Breaks the zero-sum by adding capacity instead of redistributing. The second opinion said this is a crowded space (O-LoRA, C-LoRA, etc.) but it might be the only thing that actually breaks the wall at r=32. Not tried.

---

## The question for the second opinion

**Is the user's goal achievable at r=32 on 350M?**

The goal: ACC ≥ 0.35 AND BWT ≤ −0.02 (both, simultaneously, on seed 123).

Everything we've tried either gets ACC up (gradient repair, 0.41-0.43) or BWT up (interpolation, −0.002) but not both. The zero-sum wall at r=32 seems to be the binding constraint. The question:

1. Is there a repair mechanism we haven't tried that breaks the wall (not navigates it)?
2. Is the 10-step interpolation cap actually the pragmatic answer (ACC~0.35, BWT~−0.02) and we're overthinking this?
3. Is the correction adapter (additive capacity) worth trying despite being a crowded space?
4. Or is the honest answer "AVR at r=32 on 350M can't break the wall, and the shipped claim should be 'beats SLAO+MVA on ACC with competitive BWT' instead of 'maintains BWT while raising ACC'"?

---

## Environment

- Model: LiquidAI/LFM2.5-350M (hybrid conv+attn, 16 layers)
- LoRA: rank 32, alpha 32, targets = [in_proj, out_proj] (26 modules)
- Benchmark: TRACE (C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds), 5000-example variant
- Hardware: Kaggle T4 (16GB)
- Training: 3 epochs, lr 2e-4, batch 8, ctx 512
- Repair: lr 1e-4, λ 0.5→0.1, max 50 steps, bidirectional stop at 1.15×
- Baselines: Naive (no CL), SLAO+MVA (ICLR 2026)
