# Worklog — Tiny-CL Project

---
Task ID: 1
Agent: main
Task: V9 SLAO controlled experiment script

Work Log:
- Reviewed V9 SLAO implementation (v9_slao_correct.py)
- Identified that hardcoded comparison numbers (naive=1.64x, V6=1.52x, etc.) are from different scripts/configs — fake comparison
- Verified module targeting: target_modules=["in_proj","out_proj"] hits 26 layers = 20 conv + 6 attn out_proj, NOT conv-only
- CONV_LAYER_IDS and ATTN_LAYER_IDS defined but never used for filtering
- V9 rank=32 Kaggle result: FF(A)=1.09x, FF(B)=1.06x — good numbers but comparison is unearned
- Created v9_slao_controlled.py with fixes:
  1. Naive baseline in same run (same data, same seed, same model)
  2. Module verification: prints what LoRA actually wraps (conv/attn/other counts)
  3. Multi-seed (3 seeds: 42, 123, 456)
  4. Honest naming (no "conv_only" misnomer)
  5. Aggregated FF with mean/std across seeds
  6. No hardcoded comparison numbers

Stage Summary:
- V9 rank=32 got FF(A)=1.09x on single seed — promising but needs controlled verification
- Key finding: LoRA is hitting attention out_proj too (26 layers, not 20) — this is actually fine for performance but the "conv-only" framing is wrong
- Script saved to /home/z/my-project/tiny-cl/v9/v9_slao_controlled.py
- Next: run on Kaggle to get honest comparison

---
Task ID: 2
Agent: main
Task: V9.1 controlled comparison results + V11 combo design

Work Log:
- V9.1 Kaggle results CONFIRMED: naive FF(A)=1.517±0.020, SLAO FF(A)=1.097±0.007
- SLAO reduces forgetting by 27.7% vs naive (honest same-run comparison)
- Old hardcoded "naive=1.64x" was wrong; real naive=1.52x
- SLAO stable across 3 seeds (std=0.007)
- Subtle plasticity cost: SLAO's newest-domain PPL ~0.3 worse than naive
- Designed V11 combo experiment addressing all 6 checklist items:
  1. AVR repair target = merged_state snapshot (constant memory, not per-domain)
  2. Repair fires AFTER SLAO merge (clean causal story)
  3. Repair fire count tracked (zero is reportable)
  4. EWC baseline added (finally)
  5. Compute-matched naive_ext (extra steps = avg AVR repair steps)
  6. Forward + reverse domain order
  7. Plasticity cost reported for every method
- Methods: naive, ewc, slao, slao_avr, naive_ext
- 3 seeds × forward + 1 seed × reverse = ~16 runs total

Stage Summary:
- V9.1 earned result: SLAO FF(A)=1.097±0.007 is real
- V11 script saved to /home/z/my-project/tiny-cl/v11/v11_combo.py
- Key design decision: AVR is component (c), not pivot or replacement
- If AVR never fires on top of SLAO, that's a clean finding: methods are redundant

---
Task ID: 3
Agent: main
Task: V11 combo results + V13 full picture design

Work Log:
- V11 Kaggle results confirmed:
  - Forward order: AVR fires 0 times on top of SLAO (safety net, not primary)
  - SLAO FF(A)=1.097 holds in forward, ~1.15x in reverse (Creative first is harder)
  - EWC FF(A)=1.33x (mediocre, Fisher diagonal too coarse for hybrid blocks)
  - AVR is redundant in easy cases, helps in hard cases
- Designed V13 to answer "where does the 5% gap to <1.05x come from?"
- Tested 5 hypotheses with 5 methods in same run:
  1. Cross-term noise (B_old@A_new) → Fixed-A test
  2. Capacity bottleneck → SLAO r=64 test
  3. B interpolation dilution → ΔW-stitch in product space
  4. Reproducibility check → SLAO r=32 re-run
  5. Curriculum bias → 5 domain orderings

Stage Summary:
- V11 results: SLAO is the winner, AVR is a safety net
- V13 designed to diagnose the 5% gap and pick the final shipped method

---
Task ID: 4
Agent: main
Task: V13 results + final ship decision + repo push

Work Log:
- V13 Kaggle results (3 seeds, forward A→B→C, all 5 methods):
  - naive     FF(A)=1.517±0.020 (floor, ~50% forgetting)
  - slao      FF(A)=1.097±0.007 (shipped)
  - fixed_a   FF(A)=1.149±0.005 (WORSE than SLAO — cross-term noise NOT the bottleneck)
  - dw_stitch FF(A)=207.768±54.980 (BLEW UP — additive B_cat bug, kept as negative result)
  - slao_r64  FF(A)=1.091±0.015 (barely better than r=32 — capacity NOT the bottleneck)
- Domain ordering (seed=42, SLAO):
  - A→B→C: 1.09/1.06 (baseline)
  - C→B→A: 1.19/1.03 (Creative first hurts Medical most)
  - A→C→B: 1.03/1.05 (best curriculum)
- Diagnosis: bottleneck is B interpolation dilution (Medical at 12.4% weight after 3 tasks)
- ΔW-stitch in product space is the mathematically correct fix but has additive B_cat bug;
  interpolative fix [(1-λ)·B_old, λ·B_new] not re-tested
- User decision: "SLAO it is and we ship that as the living model"
- No more untested paths. SLAO r=32 is the shipped method.

Stage Summary:
- Continual learning core: SHIPPED. SLAO r=32 on LFM2.5-350M, FF(A)=1.097±0.007.
- The 5% gap to <1.05x target is understood and accepted.
- Files pushed to GitHub:
  - tiny-cl/v13/v13_full_picture.py (the script)
  - tiny-cl/v13/v13_kaggle_output.txt (raw Kaggle output)
  - tiny-cl/v13/README.md (results + decision)
  - README.md (project root, full state)
  - NEXT_STEPS.md (self-improvement plan, 6-week sprint)
- Next phase: self-improvement without forgetting (SDFT + binary feedback + SLAO + sub-1B LoRA).
  Builds on top of SLAO core, does not modify it.
