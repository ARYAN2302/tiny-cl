# avr-cl

**Your fine-tune silently broke your model. avr-cl tells you, and fixes it.**

<p align="center">
  <img src="results/qwen3_1.7b/validation_heatmap_math.png" width="800">
</p>

*Left: naive sequential SFT — prior tasks collapse (GSM8K 66% → 9%). Right: avr-cl — prior tasks survive (GSM8K 62% → 47%). Same model, same data, same LoRA. The only difference: avr-cl watches for drift and repairs it.*

## The problem

Every "continual learning" method in LLMs is just continual absorption with weight updates. Absorb new data → update weights → try not to forget. EWC, replay, SLAO, AVR — all variations of the same thing: absorb → update → hope.

But that's not learning. When a human learns something new, they don't just absorb it and move on. They absorb it, then **verify** it against what they already know — does this break something I learned before? If it does, they **repair** the conflict — not by suppressing the new knowledge, but by integrating it properly. Then they check again. Only when the old knowledge still holds do they call it "learned."

Real learning is: **absorb → verify → repair → call it learned.**

Current CL does: **absorb → call it learned.**

The verify and repair steps are missing. That's why fine-tuning silently breaks models — nobody checked.

avr-cl builds those missing steps as a post-training phase:

- **A** = Anchor — snapshot the model before learning, so you have something to verify against
- **V** = Verify — check old tasks after absorption. Did anything break?
- **R** = Repair — fix what broke. Closed-form weight interpolation, no gradients, no old data

Every major post-training framework (TRL, Unsloth, Axolotl, LEAP) trains one stage at a time. None of them detect or repair the forgetting that happens between stages. They just move on and hope for the best.

avr-cl is the missing layer.

## What it does

After each training stage, avr-cl:

1. **VERIFY** — computes PPL on prior tasks' held-out data. If `PPL_now / PPL_best > 1.15`, the model has drifted. It forgot.
2. **REPAIR** — pulls LoRA weights back toward a snapshot: `θ ← (1−α)·θ + α·θ_snapshot`. Closed-form math. No gradients. No replay data. Runs in milliseconds.
3. **SNAPSHOT** — saves the current LoRA state so the next stage can repair toward it.

No replay buffer. No old training data. One LoRA snapshot (same size as the adapter). Constant memory regardless of how many tasks you've trained on.

## Results

### Qwen3-1.7B — Math reasoning

4-task stream: GSM8K → MATH(algebra) → AQuA-RAT → SVAMP. LoRA r=128, 5000 examples/task.

| | Naive SFT | avr-cl | What happened |
|---|---|---|---|
| **BWT** | −0.453 | **−0.078** | 5.8× less forgetting |
| **GSM8K after all 4 tasks** | 9% | **47%** | Task 1 survived |
| **ACC** | 0.220 | **0.529** | 2.4× higher overall |
| **Repair steps** | — | 29 | Drift caught and fixed 29 times |

Naive SFT collapses — by the end, the model only knows the last task. avr-cl retains all four. The repair loop fired 29 times: each time it detected PPL drift on prior tasks and pulled weights back until the drift resolved.

Full results + R-matrix: [`results/qwen3_1.7b/`](results/qwen3_1.7b/)

### LFM2.5-230M — Intent classification (cross-architecture)

4-task stream: Banking77 → CLINC150 → SNIPS → Emotion. LoRA r=128 on both conv + attention blocks.

| | Naive SFT | avr-cl |
|---|---|---|
| **BWT** | −0.112 | **+0.012** |
| **Repair steps** | — | 20 |

Positive backward transfer on a hybrid architecture — the model got *better* at old tasks by learning new ones. avr-cl works on both pure transformers (Qwen3) and hybrid conv+attention models (LFM2). Preliminary (scorer needs refinement). Full log: [`results/lfm2_230m/`](results/lfm2_230m/)

## Install

```bash
pip install git+https://github.com/ARYAN2302/tiny-cl.git
```

## Quickstart

```bash
avr train avr/configs/trace_lfm350m.yaml
```

```python
from avr import AVRStrategy
from avr.framework import ContinualPostTrainer
from avr.detectors import PPLRatioDetector
from avr.operators import SnapshotInterp
from avr.trainer import SFTStrategy
from avr.data import load_trace

trainer = ContinualPostTrainer(
    learn=SFTStrategy(epochs=3, lr=2e-4),
    verify=PPLRatioDetector(threshold=1.15),
    repair=SnapshotInterp(alpha=0.1, max_steps=10),
)
tasks = load_trace(output_dir="data")
state = trainer.run_stream(model, tokenizer, tasks)
print(f"BWT: {state.metrics['BWT']:.3f}  Repairs: {state.total_repair_steps}")
```

## How it works

```
For each task in a sequential stream:

  LEARN     → fine-tune on the new task (SFT, any LoRA config)
  VERIFY    → compute PPL on prior tasks' held-out data
              if PPL_now / PPL_best > 1.15 → drift detected
  REPAIR    → θ ← (1−α)·θ + α·θ_snapshot  (closed-form, no gradients)
              repeat until drift resolves or 10-step cap
  SNAPSHOT  → save current LoRA state for next task's repair target
```

## Why not just...?

| Approach | Problem |
|---|---|
| **Replay buffers** | Need to store old training data. Privacy, plumbing, maintenance. |
| **Retrain from scratch** | Expensive. Days of compute for every new task. |
| **LoRA adapter switching** | Needs a router at inference. Multiple adapters in memory. Doesn't compose. |
| **mergekit** | Merges N separately-trained models *after* the fact. avr-cl prevents the damage *during* one training run. |
| **Just use TRL** | TRL has no concept of "after this task, before the next." It doesn't know your model forgot. |

avr-cl needs zero old data, zero gradients at repair time, one LoRA snapshot in memory, and it *knows* when the model forgot.

## Config

```yaml
model:
  id: LiquidAI/LFM2.5-350M
  lora_targets: [in_proj, out_proj]
  lora_rank: 32

stream:
  benchmark: trace
  tasks: [C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds]
  seed: 42

learn:
  method: sft
  epochs: 3
  lr: 2e-4

verify:
  detector: ppl_ratio
  threshold: 1.15        # fire repair if PPL > 1.15× best

repair:
  operator: snapshot_interp
  alpha: 0.1             # pull strength per step
  max_steps: 10
```

## Repository

```
tiny-cl/
├── avr/                        # The framework — 1600+ lines of real code
│   ├── framework.py            # LEARN → VERIFY → REPAIR orchestrator
│   ├── detectors.py            # PPLRatioDetector (VERIFY)
│   ├── operators.py            # SnapshotInterp (REPAIR)
│   ├── trainer.py              # SFTStrategy (LEARN)
│   ├── strategy.py             # Config → trainer builder
│   ├── metrics.py              # R-matrix, BWT, FF, ACC
│   ├── data.py                 # Data loaders (TRACE, MMLU, real-world)
│   ├── cli.py                  # `avr train config.yaml`
│   ├── configs/                # YAML configs
│   └── pyproject.toml          # pip install avr-cl
├── scripts/                    # Validation experiments
│   ├── avr_cl_math_qwen3_1.7b.py       # Qwen3-1.7B math stream (headline)
│   ├── avr_cl_lfm230m_intent.py         # LFM2.5-230M intent stream (cross-arch)
│   └── validate_single_finetune.py      # Single-fine-tune validation
└── results/                    # Results + heatmaps
    ├── qwen3_1.7b/
    └── lfm2_230m/
```

## Limitations

- **Validated on 230M–1.7B.** 7B+ is next.
- **SFT only.** DPO/GRPO on the roadmap.

## License

MIT.
