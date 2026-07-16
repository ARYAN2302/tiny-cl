# avr-cl

**Your fine-tune silently broke your model. avr-cl checks if it broke, and fixes it.**

The forgetting-prevention layer for LLM post-training. After each fine-tuning stage, avr-cl detects if the model forgot prior tasks and repairs the damage in weight space — no replay buffer, no old training data, one LoRA snapshot in memory.

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
pip install avr-cl
```

## Use it

**Option 1: Full loop** — avr-cl handles everything (model loading, LoRA, SFT, verify, repair, eval):

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

**Option 2: As a layer** — keep your existing training loop (TRL, Axolotl, Unsloth), add avr-cl between stages:

```python
import avr
from trl import SFTTrainer  # your existing training

# Train on task A (your existing code)
trainer = SFTTrainer(model, train_dataset=task_a)
trainer.train()

# After training: check if the model forgot prior tasks
snapshot = avr.get_lora_state(model)  # snapshot before training task B
# ... train on task B ...
drift = avr.check_drift(
    current_ppls=avr.eval_ppls(model, tokenizer, prior_tasks, ...),
    best_ppls=best_ppls,
    completed_tasks=prior_task_names,
    threshold=1.15,
)
if drift:
    avr.repair(model, snapshot, alpha=0.1)  # 2 lines, plugs into anything
```

The layer API: `avr.get_lora_state()`, `avr.check_drift()`, `avr.repair()`. Use them with any training framework.

Each task is a `(name, train_examples, eval_examples)` tuple. Each example is a `(question, answer, gold)` triple:
- `question` — the input prompt
- `answer` — the full training target (reasoning + answer)
- `gold` — the short answer for scoring

The full-loop API handles: model loading, LoRA, chat templates, SFT training, PPL drift detection, weight repair, batched evaluation, R-matrix, BWT/FF/ACC.

## The framework

Each phase is a separate module. Use the defaults, or swap your own:

```python
from avr.learn import train_sft       # LEARN: your training function
from avr.verify import check_drift     # VERIFY: your drift detector
from avr.repair import repair          # REPAIR: your repair operator
from avr.eval import evaluate          # evaluation + scoring
```

```
avr/
├── model.py     — load_model, chat template handling
├── learn.py     — train_sft, consolidate (two-stream)
├── verify.py    — compute_ppl, check_drift
├── repair.py    — get/set/reset LoRA state, weight interpolation
├── eval.py      — batched generation, scoring, R-matrix
├── run.py       — orchestrator: wires LEARN → VERIFY → REPAIR
└── cli.py       — avr train config.yaml
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
| **EWC** | Fisher penalty slows forgetting but doesn't prevent it. Barely better than Naive (see LFM2.5 results above). |
| **mergekit** | Merges N separately-trained models *after* the fact. avr-cl prevents the damage *during* one training run. Complementary, not competing. |
| **TRL / Axolotl / Unsloth** | Great training frameworks — but none of them check if your model forgot between stages. avr-cl plugs into them as a layer. |
| **Letta / memory layers** | Handles the retrieval/memory layer for agents. avr-cl handles the weight layer. Use both. |

avr-cl needs zero old data, zero gradients at repair time, one LoRA snapshot, and it *knows* when the model forgot. It's not a replacement for your training framework — it's the layer that watches for forgetting between stages.

## Reproduce the headline result

```bash
# On Kaggle T4 or any GPU with 16GB+ VRAM
python scripts/avr_cl_math_qwen3_1.7b.py
```

This script is standalone and reproduces the Qwen3-1.7B math stream results shown above. The `avr.run()` API implements the same logic in a pip-installable package.

## Limitations

- **Validated on 1.7B and 1.2B.** 7B+ is on the roadmap.
- **SFT only.** DPO/GRPO support is the next milestone — the 2026 post-training frontier.
- **Repair cap.** `max_repair_steps=10` sometimes hits before full convergence on hard transitions. Bumping to 20 may help on difficult task streams.

## Roadmap

- [x] PyPI release (`pip install avr-cl`)
- [ ] Quickstart Colab notebook
- [ ] HuggingFace Hub integration (`avr.push_to_hub`)
- [ ] DPO/GRPO support for the LEARN phase
- [ ] 7B+ model validation
- [ ] arXiv preprint

## License

MIT.
