# avr-cl

**Your fine-tune silently broke your model. avr-cl tells you, and fixes it.**

<p align="center">
  <img src="results/qwen3_1.7b/validation_heatmap_math.png" width="800">
</p>

*Left: naive sequential SFT — prior tasks collapse (GSM8K 66% → 9%). Right: avr-cl — prior tasks survive (GSM8K 62% → 47%). Same model, same data, same LoRA.*

## The problem

Every continual learning method in LLMs is just absorption with weight updates. Absorb new data → update weights → try not to forget. EWC, replay, SLAO — all variations of the same thing: absorb → update → hope.

But that's not learning. When a human learns something new, they don't just absorb it and move on. They absorb it, then **verify** it against what they already know — does this break something I learned before? If it does, they **repair** the conflict. Then they check again. Only when the old knowledge still holds do they call it "learned."

Real learning is: **absorb → verify → repair → call it learned.**

Current CL does: **absorb → call it learned.**

The verify and repair steps are missing. That's why fine-tuning silently breaks models — nobody checked.

avr-cl builds those missing steps:

- **Anchor** — snapshot the model before learning
- **Verify** — check old tasks after absorption. Did anything break?
- **Repair** — fix what broke. Closed-form weight interpolation, no gradients, no old data

## Results

Qwen3-1.7B, 4-task stream: GSM8K → MATH(algebra) → AQuA-RAT → SVAMP. LoRA r=128, 5000 examples/task.

| | Naive SFT | avr-cl |
|---|---|---|
| **BWT** | −0.453 | **−0.078** |
| **GSM8K after all 4 tasks** | 9% | **47%** |
| **ACC** | 0.220 | **0.529** |
| **Repair steps** | — | 29 |

5.8× less forgetting. The repair loop fired 29 times across 3 tasks — each time detecting PPL drift on prior tasks and pulling weights back until the drift resolved.

Results: [`results/qwen3_1.7b/`](results/qwen3_1.7b/)

## Install

```bash
pip install git+https://github.com/ARYAN2302/tiny-cl.git
```

## Use it

```python
import avr

result = avr.run(
    model="Qwen/Qwen3-1.7B",
    tasks=[
        ("task_a", train_examples, eval_examples),
        ("task_b", train_examples, eval_examples),
    ],
    lora_rank=128,
)

print(f"BWT: {result['bwt']:+.3f}  Repairs: {result['repairs']}")
```

Each task is a `(name, train_examples, eval_examples)` tuple. Each example is a `(question, answer, gold)` triple:
- `question` — the input prompt
- `answer` — the full training target (reasoning + answer)
- `gold` — the short answer for scoring

The framework handles: model loading, LoRA, chat templates, SFT training, PPL drift detection, weight repair, batched evaluation, R-matrix, BWT/FF/ACC.

Or via CLI:

```bash
avr train config.yaml --seed 42
```

Config format:
```yaml
model: Qwen/Qwen3-1.7B
lora_rank: 128
tasks:
  - name: task_a
    train: data/task_a_train.json
    eval: data/task_a_eval.json
  - name: task_b
    train: data/task_b_train.json
    eval: data/task_b_eval.json
```

## How it works

```
For each task in a sequential stream:

  LEARN     → fine-tune on the new task (SFT, any LoRA config)
  VERIFY    → compute PPL on prior tasks' data
              if PPL_now / PPL_best > 1.15 → drift detected
  REPAIR    → θ ← (1−α)·θ + α·θ_snapshot  (closed-form, no gradients)
              repeat until drift resolves or 10-step cap
  SNAPSHOT  → save current LoRA state for next task's repair target
```

No replay buffer. No old training data. One LoRA snapshot in memory.

## Why not just...?

| Approach | Problem |
|---|---|
| **Replay buffers** | Need to store old training data. Privacy, plumbing, maintenance. |
| **Retrain from scratch** | Expensive. Days of compute for every new task. |
| **LoRA adapter switching** | Needs a router at inference. Multiple adapters in memory. |
| **mergekit** | Merges N separately-trained models *after* the fact. avr-cl prevents the damage *during* one training run. |
| **Just use TRL** | TRL has no concept of "after this task, before the next." It doesn't know your model forgot. |

## Reproduce the headline result

```bash
# On Kaggle T4 or any GPU with 16GB+ VRAM
python scripts/avr_cl_math_qwen3_1.7b.py
```

This script is standalone (doesn't require the avr package) and reproduces the Qwen3-1.7B math stream results shown above. The `avr.run()` API in the package implements the same logic.

## Limitations

- **Validated on 1.7B.** Smaller and larger models are next.
- **SFT only.** DPO/GRPO on the roadmap.

## License

MIT.
