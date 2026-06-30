# V15: SLAO + MVA — 3-Round Loop with Threshold Tracking

Answers the question v14 couldn't: does pass@5 trend up across rounds 2 and 3,
or does it bounce around near +0.07 in a way that looks like noise?

v14 showed MVA composes with SLAO (+0.070 pass@5, 1 round). But χ²=1.8
vs 3.84 needed means 1 round is close to a coin flip. v15 runs 3 MVA rounds
to see if the signal compounds.

## Architecture

```
Rounds 1-3: Standard v13 SLAO (train domains A→B→C, SLAO merge after each)
Rounds 4-6: MVA self-improvement rounds (certainty-validated SQuAD, SLAO merge)
```

Each MVA round:

1. Generate answers on FRESH SQuAD sample (different questions each round)
2. Compute certainty distribution (LOGGED — does it shift across rounds?)
3. Adaptive threshold (50th percentile of THIS round's distribution)
4. Validate, train, SLAO merge
5. Eval: pass@5 + domain perplexity

## What v15 answers (that v14 didn't)

- Does pass@5 trend up monotonically (signal) or bounce (noise)?
- Does the certainty distribution shift across MVA rounds?
- Does validation precision hold or degrade?
- Does domain PPL drift accumulate or stabilize?
- Is the single-round +0.07 real, or was it a coin flip?

## Verdict outputs

- **COMPOUNDS:** pass@5 gained >0.10 over 3 rounds, monotonic → build Architecture C
- **TRENDS UP:** pass^5 gained >0.05 but not monotonic → signal real but noisy
- **MARGINAL:** pass^5 gained <0.05 → single-round was mostly noise
- **DOES NOT COMPOUND:** pass^5 declined or flat → fundamental integration issue

## Config

Same as v14:

- Model: LiquidAI/LFM2.5-350M
- LoRA: rank=32, targets=`["in_proj", "out_proj"]` (v13 minimal config)
- Domains: medical, code, creative (1M tokens each, 1 epoch)
- MVA: SQuAD, 200 questions/round, 3 epochs, adaptive threshold (50th percentile)
- SLAO: A=replace, B=interpolate (λ=1/√task_num)

## Runtime

~4-5 hours on T4 (3 domain rounds ~2.5h + 3 MVA rounds ~2h)

## Files

- `v15_slao_mva_3round.py` — The 3-round loop script. One file, one Kaggle cell.

## Framing (unchanged from v14)

Self-improvement via **reduced-error filtering**, not self-improvement via correct
self-generated data. MVA's certainty gate filters out 60–88% of wrong answers
(precision 79–94%), reducing wrong-answer contamination from ~30% (naive) to
~6–21% (MVA). The pass^5 improvement reflects training on fewer mistakes.

## Usage

```bash
python v15_slao_mva_3round.py
# ~4-5 hours on a single T4 GPU
```
