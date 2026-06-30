# V14: SLAO + MVA Integration

Tests whether MVA self-improvement composes with SLAO continual learning on
Liquid LFM2.5-350M. Architecture A: MVA as a 4th SLAO task on top of v13's
existing 3-domain loop.

## Architecture

```
Round 1: Train medical (1M tokens)  → SLAO merge → eval
Round 2: Train code (1M tokens)     → SLAO merge → eval
Round 3: Train creative (1M tokens) → SLAO merge → eval
Round 4: MVA self-improvement       → SLAO merge → eval  ← THE TEST
```

Round 4 measures two things, before and after MVA:
- pass@5 on SQuAD holdout — did self-improvement work?
- Perplexity on all 3 domains — did MVA break domain knowledge?

## Framing (important)

Self-improvement via **reduced-error filtering**, not self-improvement via correct
self-generated data.

MVA's certainty gate filters out 60–88% of wrong answers (precision 79–94%
across seeds), reducing wrong-answer contamination from ~30% (naive) to ~6–21%
(MVA). The pass@5 improvement reflects the model training on fewer mistakes,
not on verified-correct self-generated data.

This is the claim that survives reading the code.

## Files

- `v14_slao_mva.py` — The integration script. One file, one Kaggle cell, one run.
  ~3 hours on T4.
- `self_improvement_validation.py` — The standalone MVA validation script (v5).
  Used to validate the certainty signal on the base model before integration.
  Two-seed confirmation: MVA beats naive by +11–12pp on pass@5.

## Config

- Model: LiquidAI/LFM2.5-350M (hybrid: 6 attention + 10 conv layers)
- LoRA: rank=32, alpha=32, targets=`["in_proj", "out_proj"]` (v13's minimal config)
- Domains: medical, code, creative (1M tokens each, 1 epoch)
- MVA: SQuAD, 200 questions, certainty threshold = ADAPTIVE (50th percentile)
- SLAO: QR-decompose A → orthogonal basis; A=replace, B=interpolate (λ=1/√task_num)

## v14 result (seed 42)

```
Round            A (med)      B (code)     C (creat)    pass^5
--------------------------------------------------------------
R1 (A)           14.59        5.71         10.00        0.330
R2 (B)           14.95        3.80         8.48         0.350
R3 (C)           16.20        3.93         5.96         0.450
R4 (MVA)         15.52        3.95         6.28         0.520

MVA round deltas:
  pass^5: 0.450 -> 0.520 (+0.070)
  PPL(A): 16.20 -> 15.52 (-0.67, improved)
  PPL(B): 3.93 -> 3.95 (+0.02, flat)
  PPL(C): 5.96 -> 6.28 (+0.32, small spike)

Validation: 103/200 validated, precision 84.5%, 16 wrong answers in training (15.5%)
```

Verdict: **COMPOSES** — MVA improves pass@5 without spiking domain perplexity.

## Caveats

1. **Modest gain.** +0.070 pass@5 = 7 questions out of 100. v5 on base model
   was +0.120. The post-SLAO model has less room to improve.

2. **Not statistically significant.** McNemar estimate: χ² = 1.8 (need >3.84
   for p<0.05). Directional, not conclusive. One seed, one run.

3. **Adaptive threshold was necessary.** The fixed 17.0 threshold (tuned on base
   model in v5) produced 0/200 validated pairs on the post-SLAO model. SLAO
   shifted the certainty distribution: base mean=16.68 → post-SLAO mean=11.58.
   The adaptive threshold (50th percentile) made the filter operational.

4. **Known flaw: 15.5% wrong answers in training.** The certainty gate filters
   out most wrong answers but lets confident-wrong ones through. This is the
   "reduced-error filtering" framing — not "correct self-generated data."

## What v14 answers

- Does MVA's certainty signal survive SLAO merging? **Yes** (84.5% precision)
- Does MVA's update compose with SLAO's merged state? **Yes** (no PPL spike)
- Does MVA improve pass@5 on the SLAO-merged model? **Yes** (+0.070, directional)

## What v14 does NOT answer

- Is +0.070 real or noise? (Need multi-round trajectory, not single round)
- Does the gain compound over multiple MVA rounds? (Need Architecture C)
- Does the adaptive threshold need recalibration each round? (Need to track it)
- What's the noise floor per round? (Need 3+ rounds to estimate)

## Next step: v15

3-round loop with threshold tracking. Cheaper than full 10-round Architecture C,
more informative than another seed of v14. Directly answers: does pass@5 trend
up across rounds 2 and 3, or does it bounce around near +0.07 in a way that
looks like noise?

## Usage

```bash
python v14_slao_mva.py
# ~3 hours on a single T4 GPU
```
