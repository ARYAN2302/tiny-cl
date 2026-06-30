# Experiments — Version Index

Every iteration from V1 (30M GPT-2 from scratch) through V23 (AVR standalone on TRACE). Versions that we skipped (v1, v12, v16, v17, v19, v20, v21, v22) were either superseded the same day or never made it past a whiteboard.

---

## V1 — Anchor-AVR on 30M GPT-2 (from scratch)

**Goal:** Prove that anchor-pull self-correction works at all, on a small model trained from scratch.

**Mechanism:** Train GPT-2 (30M params) sequentially on Stories → Wiki → News. After each phase, check old-domain perplexity; if it drifted, pull all weights back toward an anchor snapshot via `θ ← (1−α)·θ + α·θ_anchor`.

**Result:** 3× less forgetting than naive sequential SFT and than EWC.

**Files:** [`docs/BUILD.md`](BUILD.md) (full V1 build log).

**Status:** Superseded by V2 (move to pretrained LFM2.5-350M). Kept as the origin story for AVR.

---

## V2 — Migration to LFM2.5-350M + LoRA

**Goal:** Take a pretrained sub-1B model and make it continual. The industry-relevant version of V1.

**Mechanism:** Freeze LFM2.5-350M base, attach LoRA adapters, apply Anchor-AVR on the LoRA weights only (not the base weights). Three domains: Medical / Code / Creative, 1M tokens each.

**Files:** The V2 framework under [`../tiny-cl/v2/`](../tiny-cl/v2/) (`config.py`, `data.py`, `methods.py`, `train.py`, `evaluate.py`, `model.py`, `run_all.py`, `plot_results.py`, `modal_run.py`). Full build log in [`BUILD_V2.md`](BUILD_V2.md).

**Status:** Superseded by the single-file Kaggle scripts (v4 onward). Kept because it's the cleanest reference implementation of the V2 framework.

---

## V3 — Method tuning

**Goal:** Tune the Anchor-AVR method on the V2 framework. Compare against EWC, Replay, Freeze.

**Files:** [`tiny-cl/v3/`](../tiny-cl/v3/) — mirrors V2 layout.

**Status:** Superseded.

---

## V4 — First Kaggle port

**Goal:** Move V2 to a single Kaggle cell so it runs on free T4 GPUs.

**Files:** [`tiny-cl/v4/`](../tiny-cl/v4/) — `v4_kaggle.py`, `BUILD.md`, V2-style Python files.

**Status:** Superseded by V5+ (cleaner single-file scripts).

---

## V5 — MVA validation on SQuAD (standalone)

**Goal:** Validate that the INTUITOR certainty signal correlates with correctness on LFM2.5-350M, and that a certainty-gated self-improvement loop beats naive SFT on pass@5.

**Mechanism:** Generate 5 answers per SQuAD question, compute `KL(U‖p)` certainty, filter at threshold τ=17.0 (tuned on base model), train on filtered set, evaluate pass@5.

**Result:** MVA beats naive by +12pp on pass@5 (seed 42). Certainty signal `r=0.341` with correctness (p<0.001).

**Files:** [`tiny-cl/v5/v5_kaggle.py`](../tiny-cl/v5/). Seed-123 confirmation in [`tiny-cl/v14/self_improvement_validation.py`](../tiny-cl/v14/self_improvement_validation.py).

**Status:** Shipped (the MVA mechanism is reused in v14/v15 on top of SLAO).

---

## V6 — Diagnostics

**Goal:** Diagnose why V5's certainty gate lets confident-wrong answers through (the 6–21% wrong-answer contamination).

**Files:** [`tiny-cl/v6/`](../tiny-cl/v6/) — `v6_diagnostic.py`, `v6_kaggle.py`, `v6_kaggle_notebook.ipynb`.

**Status:** Diagnostic; results fed into v14's adaptive threshold.

---

## V7 — Kaggle baseline

**Files:** [`tiny-cl/v7/v7_kaggle.py`](../tiny-cl/v7/).

**Status:** Superseded.

---

## V8 — O-LoRA attempt

**Goal:** Try O-LoRA (orthogonal LoRA, from the ICLR 2025 O-LoRA paper) as an alternative to Anchor-AVR.

**Bug:** V8 only saved/restored conv-layer LoRA, missing attention-layer LoRA. Fixed in V8.1 (`v8_olora_diagnostic.py`).

**Files:** [`tiny-cl/v8/`](../tiny-cl/v8/) — `v8_kaggle.py`, `v8_norm_test.py`, `v8_olora_diagnostic.py`.

**Status:** Superseded by V9 (SLAO is a strict improvement).

---

## V9 — First honest SLAO vs naive

**Goal:** Run SLAO and naive in the *same* script, same data, same seed, same model. V9.1 adds module verification (print what LoRA actually wraps) and multi-seed.

**Mechanism:** SLAO Algorithm 1 (Qiao & Mahdavi, ICLR 2026, arXiv 2512.23017):

```
1. Task 1: standard fine-tune → A_merge=A_1, B_merge=B_1
2. Task i (i>1):
   a. A_init = QR(prev_A)^T (orthogonal rows), B_init = prev_B
   b. Fine-tune both A and B on new task
   c. A_merge = A_ft (replace), B_merge = B_merge + λ(B_ft - B_merge)
      where λ = 1/sqrt(i)
```

**Result:** FF(A)=1.09× with rank=32, vs naive FF(A)≈1.50×.

**Files:** [`tiny-cl/v9/v9_slao_controlled.py`](../tiny-cl/v9/), `v9_slao_correct.py`.

**Status:** Superseded by V13 (full picture, multi-seed, multi-ordering).

---

## V10 — SLAO + SVD digest

**Goal:** After SLAO merge, recompress the LoRA product `B·A` via SVD to keep only the top-r singular directions. The model "digests" after each meal — absorbs what matters, drops what doesn't.

**Mechanism:**

```
1. Train → SLAO merge (A=replace, B=interpolate)
2. SVD: B@A = U@S@V^T → keep top-r → new B, new A
3. A now has orthogonal rows (free ortho init for next task!)
```

**Files:** [`tiny-cl/v10/v10_living_model.py`](../tiny-cl/v10/).

**Status:** Intermediate idea; SVD digest did not measurably beat plain SLAO. The "free ortho init" insight was folded into V13.

---

## V11 — SLAO + AVR combo

**Goal:** Combine SLAO merge with AVR verify-repair. Add EWC baseline and reverse-order test.

**Mechanism:** SLAO merge after each task, then AVR verify-repair fires if any prior task's PPL drifted >1.15× above its best.

**Key design decision:** AVR repair target = previous `merged_state` snapshot (constant memory), not the per-task fine-tuned state. Repair fires AFTER SLAO merge (clean causal story). Track repair fire count (zero is a real, reportable result).

**Files:** [`tiny-cl/v11/v11_combo.py`](../tiny-cl/v11/).

**Status:** Shipped (the AVR mechanism is reused in v23 standalone).

---

## V13 — The Full Picture (shipped CL core)

**Goal:** Final continual-learning experiment. Five methods, five domain orderings, multi-seed, with cross-term noise measurement.

**Methods tested:**
1. **naive** — sequential SFT, no CL. Floor.
2. **slao** — SLAO Algorithm 1 (Qiao & Mahdavi, ICLR 2026). **Shipped.**
3. **fixed_a** — freeze A after task 1, train B only. Diagnostic for cross-term noise.
4. **dw_stitch** — merge in ΔW space via QR + 2r×2r SVD. Memory-efficient. **Has a bug** (additive B_cat instead of interpolative). Kept as negative result.
5. **slao_r64** — SLAO at rank 64. Capacity check.

**Result:** SLAO r=32 ships. FF(A) = 1.097 ± 0.007, FF(B) = 1.065 ± 0.005. 5.5× better retention than naive.

**Files:** [`tiny-cl/v13/v13_full_picture.py`](../tiny-cl/v13/), `v13_kaggle_output.txt` (raw Kaggle output — every method × seed × ordering).

**Status:** **Shipped.** The CL core is frozen.

---

## V14 — SLAO + 1 MVA round (integration test)

**Goal:** Test whether MVA self-improvement composes with SLAO continual learning. Architecture A: MVA as a 4th SLAO task on top of v13's existing 3-domain loop.

**Architecture:**

```
Round 1: Train medical (1M tokens)  → SLAO merge → eval
Round 2: Train code (1M tokens)     → SLAO merge → eval
Round 3: Train creative (1M tokens) → SLAO merge → eval
Round 4: MVA self-improvement       → SLAO merge → eval  ← THE TEST
```

**Result:** MVA composes with SLAO. +0.070 pass@5 (0.450 → 0.520), no PPL spike on prior domains. Directional but not statistically significant (McNemar χ²=1.8 vs 3.84 needed).

**Files:** [`tiny-cl/v14/v14_slao_mva.py`](../tiny-cl/v14/), `self_improvement_validation.py`.

**Status:** Shipped. Single-round directional result; v15 tests compounding.

---

## V15 — SLAO + 3 MVA rounds (compounding test)

**Goal:** Answer the question v14 couldn't: does pass@5 trend up across rounds 2 and 3, or does it bounce around near +0.07 in a way that looks like noise?

**Architecture:**

```
Rounds 1-3: Standard v13 SLAO (train domains A→B→C, SLAO merge after each)
Rounds 4-6: MVA self-improvement rounds (certainty-validated SQuAD, SLAO merge)
```

Each MVA round uses a fresh SQuAD sample, an adaptive threshold (50th percentile of that round's certainty distribution), and logs the certainty distribution shift across rounds.

**Files:** [`tiny-cl/v15/v15_slao_mva_3round.py`](../tiny-cl/v15/).

**Status:** Shipped.

---

## V18 — TRACE benchmark (naive vs SLAO+MVA)

**Goal:** Test the living model on the recognized continual-learning benchmark. Four tasks from TRACE that work on 350M: C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds.

**Metrics:** ACC (overall), BWT (forgetting), FWT (forward transfer). Standard GEM/NeurIPS 2017 definitions.

**Files:** [`tiny-cl/v18/v18_trace_benchmark.py`](../tiny-cl/v18/).

**Status:** Shipped.

---

## V23 — AVR standalone on TRACE

**Goal:** AVR alone on TRACE. No SLAO, no MVA. Isolate AVR's contribution.

**Algorithm:**

1. Train on task
2. Check PPL drift on previous tasks (ratio > 1.15 = drifted)
3. If drifted: repair toward snapshot via `θ ← (1−α)·θ + α·θ_snapshot`
4. Next task

**Files:** [`tiny-cl/v23/v23_avr_trace.py`](../tiny-cl/v23/).

**Status:** Shipped.

---

## Versions we skipped

| Version | Why skipped |
|---------|-------------|
| v1      | Folded into V2; the 30M from-scratch story is in `docs/BUILD.md` |
| v12     | Never made it past design — merged into v13 |
| v16, v17| Alternative MVA thresholds; superseded by v15's adaptive threshold |
| v19, v20, v21, v22 | Iterations on TRACE harness; v23 is the clean version |
