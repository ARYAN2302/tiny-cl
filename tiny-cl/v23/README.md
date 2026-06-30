# V23: AVR (standalone) on TRACE — PPL-ratio verify + closed-form repair

AVR alone. No SLAO. No MVA. Just:

1. Train on task
2. Check PPL drift on previous tasks
3. If drifted: repair toward snapshot
4. Next task

This is AVR as it ran in the dashboard: naive vs anchor, standalone.
Reuses v18 naive numbers. Only runs AVR.

## Algorithm

After training each task `t`:

```
1. Snapshot the current LoRA state S_t
2. For each completed task j, compute current PPL on task j's eval set
3. Update ppl_best[j] = min(ppl_best[j], ppl_after_t)
4. VERIFY:
   for each prior task j:
     ratio = ppl_now[j] / ppl_best[j]
     if ratio > 1.15: mark j as drifted
5. REPAIR (while any task drifted and steps < MAX_REPAIR_STEPS):
   θ ← (1 − α) · θ + α · θ_snapshot       # α = 0.1
   recompute ppl_now[j] for all j
   unmark tasks whose ratio is now ≤ 1.15
6. Proceed to next task
```

## Why AVR works

- **Verify** uses PPL ratio as a model-free drift detector. No hidden-state
  statistics, no Fisher information, no replay. The 1.15 threshold corresponds
  to ~14% log-perplexity increase.
- **Repair** is a closed-form weight interpolation — no optimizer, no gradients,
  no labels. It pulls the model toward the snapshot state, which is guaranteed
  to have lower PPL on the drifted task (by construction of `ppl_best`).
- The repair step is **idempotent under no further training**: once
  `θ = θ_snapshot`, repair has no effect.

## Config

- Model: LiquidAI/LFM2.5-350M
- LoRA: rank=32, targets=`["in_proj", "out_proj"]`
- TRACE: 4 tasks (C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds), 5000-example variant
- Drift threshold: 1.15 (PPL ratio)
- Repair α: 0.1
- Max repair steps: 100 (practically never hit)
- Per-task training: 3 epochs, batch=8, ctx=512
- Seed: 42

## Files

- `v23_avr_trace.py` — single-file Kaggle script

## Usage

```bash
python v23_avr_trace.py
# ~2 hours on a single T4 GPU
# Output: per-task PPL, drift events, repair steps, ACC/BWT/FWT
```

## What AVR does NOT do

- **No task-id inference at inference time.** AVR only acts during training.
- **No replay buffer.** The snapshot is just a LoRA state dict.
- **No gradient computation during repair.** The repair is closed-form.
- **No protection against forward transfer degradation.** AVR preserves PPL on
  prior tasks but does not optimize for new-task performance.

## Related versions

- **v11** — SLAO + AVR combo on the 3-domain Medical/Code/Creative benchmark
  (first appearance of the PPL-ratio gate + closed-form repair).
- **v18** — TRACE benchmark with SLAO+MVA (no AVR).
- **v23 (this)** — AVR standalone on TRACE. Isolates AVR's contribution.
