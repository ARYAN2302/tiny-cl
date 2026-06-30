# V4: Streaming AVR — Does Verification-Driven Repair Survive Granularity Collapse?

## The Question

Every AVR experiment so far (V1, V2, V3) used **batch-incremental** updates:
a full pass over a domain's data before moving on. But there's a harder regime
called **streaming/online continual learning** where updates arrive in tiny
increments — a handful of examples at a time.

**Does AVR's detect-and-repair mechanism still work at small increments?**

## Why This Matters

1. **Structural advantage over EWC**: EWC needs a full pass over task data
   to estimate Fisher information. At small increments (20-100 examples),
   EWC **literally cannot be computed**. AVR's anchors only need a handful
   of probe sequences — no full-pass requirement. This is a qualitative
   difference, not a quantitative one.

2. **Real-world relevance**: Gboard's 1.4M LSTM updates via federated rounds
   of ~100-500 clients' typing data, not big domain blocks. Real on-device
   learning happens in small increments.

3. **The honest risk**: Drift detection might become too noisy with few
   examples, or repair might fire too often and erase the compute advantage.
   Either outcome is reportable — "AVR degrades below increment size N" is
   a real result.

## Architecture: Two Model Scales

### Model 1: Gboard-scale LSTM (1.4M params)
- Same architecture as production Gboard neural model
- Single-layer CIFG LSTM, ~670 hidden units, embedding dim 96
- 10K vocabulary
- ~1.4MB on disk
- Training: ~2 minutes per domain on M1 Mac

### Model 2: V1-scale GPT (30M params)
- Small GPT-2 style transformer trained from scratch
- No pretrained base, no LoRA — full end-to-end training
- ~120MB in float32
- Training: ~15 minutes per domain on M1 Mac
- Same architecture as V1 (already proven to work with AVR)

## Experiment Grid

**One variable**: increment size
- full-phase (V1 baseline — already have this data)
- 500 examples
- 100 examples
- 20 examples

**Three methods** at each increment size:
- Naive (sequential, no protection)
- AVR (anchor-based verify + repair)
- EWC (where computable — flag where it isn't)

**Output**: Forgetting factor vs increment size, one line per method, one plot.

## The Expected Claim

"AVR's forgetting factor stays under X across increment sizes from full-phase
down to N examples, while naive degrades to Y at the smallest increment and
EWC is undefined below the batch size needed for a stable Fisher estimate."

## Why This Is CPU-Feasible

- 1.4M LSTM: ~5.6MB in RAM, trains in minutes, no GPU needed
- 30M GPT: ~120MB in RAM, trains in ~15 min on M1 (no pretrained download)
- No LoRA overhead (no frozen base model backward pass)
- Total grid: ~4 increment sizes × 3 methods × 2 models = 24 runs
- Estimated total: 2-4 hours on M1 Mac

## Honest Risks

1. **Anchor noise**: With 20-example increments, drift detection may fire
   randomly (false positives) or miss real drift (false negatives)
2. **Repair overhead**: If repair fires every 20 examples, total compute
   may exceed naive — AVR becomes inefficient
3. **EWC undefined**: At 20-example increments, Fisher estimation is
   meaningless — this is a feature (proves the structural advantage)
   but means one column of the grid is empty
4. **1.4M model may not forget enough**: Small LSTMs might not show
   catastrophic forgetting as clearly as transformers — the 30M GPT
   is the more reliable testbed

## File Structure

```
v4/
├── BUILD.md              # This doc
├── config.py             # All configs (models, methods, grid)
├── models.py             # LSTM (1.4M) + SmallGPT (30M)
├── data.py               # Streaming data pipeline
├── methods.py            # Naive, AVR, EWC adapted for streaming
├── anchors.py            # Anchor store adapted for small increments
├── train.py              # Main training loop
├── evaluate.py           # Perplexity + forgetting metrics
├── run_grid.py           # Run the full experiment grid
├── plot_results.py       # Forgetting vs increment size plot
└── results/              # JSON results saved here
```
