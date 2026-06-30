# V5: MVA Validation on SQuAD (standalone)

Validates the INTUITOR certainty signal on LFM2.5-350M and tests whether a certainty-gated self-improvement loop beats naive SFT on pass@5.

## What it tests

1. **Signal validity:** Does `KL(U‖p)` certainty correlate with answer correctness on LFM2.5-350M?
2. **Self-improvement:** Does training on certainty-filtered self-generated data beat naive SFT on pass@5?

## Mechanism

```
1. Sample 200 SQuAD questions (held out from training)
2. For each question, generate k=5 answers via T=0.7 sampling
3. For each answer, compute certainty = KL(U ‖ p)
4. Filter: keep only (question, answer) pairs with certainty > τ
5. Train on the filtered set (3 epochs, LoRA r=16, full 8-module config)
6. Evaluate pass@5 on a holdout of 100 questions
```

## Result (seed 42)

- **Certainty signal:** r=0.341 with correctness (p<0.001) on LFM2.5-350M
- **Validation precision:** 79–94% (the gate filters out 60–88% of wrong answers)
- **pass@5:**
  - baseline: 0.530
  - naive:    0.590
  - **MVA:    0.710** (+12pp over naive)

## Files

- `v5_kaggle.py` — single-file Kaggle script, ~90 min on T4

## Caveats

1. **Wrong-answer contamination:** 6–21% of MVA-filtered training pairs are still wrong. The certainty gate is *not* a correctness gate.
2. **Threshold sensitivity:** The fixed τ=17.0 was tuned on the base model. When MVA was applied on top of SLAO in v14, this threshold produced 0/200 validated pairs (SLAO shifts the certainty distribution). v14/v15 use an adaptive 50th-percentile threshold instead.

## Seed 123 confirmation

Run `../v14/self_improvement_validation.py` for the seed-123 confirmation. Result: MVA beats naive by +11pp on pass@5 (0.575 → 0.685). The result is robust across seeds.

## Status

Shipped. The MVA mechanism is reused in v14 and v15 on top of SLAO.
