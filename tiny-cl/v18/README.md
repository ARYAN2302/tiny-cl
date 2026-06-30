# V18: The Real Test — TRACE Benchmark with SLAO+MVA

Tests the living model on the recognized continual learning benchmark.
One file, one process, one run.

## What it does

1. Download TRACE data (500 variant, 4 tasks that work on 350M)
2. Load LFM2.5-350M + LoRA
3. Evaluate baseline on all 4 tasks (our harness, our prompts)
4. Stream tasks sequentially:
   - Method A: **NAIVE** (sequential SFT, no protection)
   - Method B: **SLAO+MVA** (the living model mechanism)

   After each task, evaluate ALL tasks seen so far
5. Build the R matrix (score on task j after training task i)
6. Compute: ACC (overall), BWT (forgetting), FWT (improvement)
7. Compare: does SLAO+MVA beat naive on both axes?

## The 4 tasks (from TRACE, verified to work on 350M)

1. **C-STANCE**  — stance classification (A/B/C)
2. **FOMC**      — finance classification (A/B/C)
3. **NumGLUE-cm** — math word problems (number)
4. **NumGLUE-ds** — math subtraction (number)

## Metrics (standard, from GEM NeurIPS 2017)

`R[i,j]` = score on task `j` after training task `i`

- **ACC** = mean of last row of `R` = overall performance after all training
- **BWT** = `mean(R[T,j] − R[j,j])` for `j<T` = how much old tasks degraded (forgetting)
  - Negative = forgetting; zero = no forgetting; positive = improvement
- **FWT** = `mean(R[i−1,i] − baseline[i])` for `i>0` = how much new tasks improved (self-improvement)

## Config

- Model: LiquidAI/LFM2.5-350M
- LoRA: rank=32, targets=`["in_proj", "out_proj"]`
- Tasks: 4 (C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds)
- TRACE variant: `LLM-CL-Benchmark_5000`
- Per-task training: 3 epochs, batch=8, ctx=512
- SLAO merge after each task (A=replace, B=interpolate)
- MVA round on SQuAD after all 4 tasks (certainty-gated, adaptive threshold)

## Runtime

~3-4 hours on T4

## Files

- `v18_trace_benchmark.py` — single-file Kaggle script

## Usage

```bash
python v18_trace_benchmark.py
# ~3-4 hours on a single T4 GPU
# Output: R matrix, ACC/BWT/FWT for naive vs SLAO+MVA
```
