# Your fine-tune silently broke your model. Here's the fix.

**Posted: July 2026**

You fine-tuned an LLM on task A. It worked great. Then you fine-tuned the same model on task B. Task B works — but task A is now broken. The model forgot.

This isn't a bug. It's catastrophic forgetting, and it happens every time you fine-tune sequentially. Nobody checks for it. The model doesn't tell you it forgot. You just ship it, and your users notice.

**avr-cl** is a forgetting-prevention layer for LLM post-training. After each fine-tuning stage, it checks whether the model forgot previous tasks — and if it did, it repairs the damage in weight space. No replay buffer. No old training data. One LoRA snapshot in memory.

```
pip install avr-cl
```

## The problem

Every continual learning method in LLMs follows the same pattern: **absorb new data → update weights → try not to forget.** EWC, replay buffers, SLAO — they all try to *prevent* forgetting during the weight update. Sometimes it works. Often it doesn't. And nobody checks afterward.

But that's not how learning works. When a human learns something new, they don't just absorb it and move on. They:
1. **Absorb** the new knowledge
2. **Verify** it against what they already know — "does this break something I learned before?"
3. **Repair** the conflict if it does
4. Only then call it "learned"

Current continual learning does step 1 and stops. Steps 2 and 3 are missing. That's why fine-tuning silently breaks models — nobody checked.

## The fix: LEARN → VERIFY → REPAIR

avr-cl adds the missing steps:

```
For each task in a sequential stream:

  LEARN     → fine-tune on the new task (SFT, any LoRA config)
  VERIFY    → compute PPL on prior tasks' data
              if PPL_now / PPL_best > 1.15 → drift detected
  REPAIR    → θ ← (1−α)·θ + α·θ_snapshot  (closed-form, no gradients)
              repeat until drift resolves or 10-step cap
  SNAPSHOT  → save current LoRA state for next task's repair target
```

The key insight: **detect forgetting, then repair it.** Not "try to prevent it and hope." After each stage, avr-cl computes the perplexity on prior tasks. If PPL drifts more than 15%, it pulls the LoRA weights back toward the snapshot until the drift resolves.

No replay buffer. No old training data. No gradients at repair time. One LoRA snapshot in memory.

## Does it work?

**Qwen3-1.7B, 4-task math stream (GSM8K → MATH → AQuA → SVAMP), 5000 examples/task:**

| | Naive SFT | avr-cl |
|---|---|---|
| **BWT** (forgetting) | −0.453 | **−0.078** |
| **GSM8K after all 4 tasks** | 9% | **47%** |
| **ACC** (final avg) | 0.220 | **0.529** |
| **Repair steps fired** | 0 | **29** |

**5.8× less forgetting.** The repair loop fired 29 times across 3 task transitions. Each time, it detected PPL drift on prior tasks and pulled weights back until the drift resolved.

The heatmap tells the story:

![avr-cl heatmap: Naive (left) collapses, avr-cl (right) preserves](https://raw.githubusercontent.com/ARYAN2302/tiny-cl/main/results/qwen3_1.7b/validation_heatmap_math.png)

*Left: naive sequential SFT — prior tasks collapse (GSM8K 66% → 9%). Right: avr-cl — prior tasks survive (GSM8K 62% → 47%). Same model, same data, same LoRA.*

## Does it generalize?

We tested on two model families, cross-domain, and the standard TRACE benchmark:

| Model / Setting | Naive BWT | avr-cl BWT | Improvement |
|---|---|---|---|
| Qwen3-1.7B (5000 ex) | −0.453 | **−0.078** | 5.8× |
| Qwen3-1.7B (500 ex) | −0.320 | **−0.037** | 8.6× |
| LFM2.5-1.2B (500 ex) | −0.280 | **−0.150** | 1.9× |
| Cross-domain (Code→Math→Instruct→Science) | — | **−0.010** | near-zero forgetting |
| TRACE 8-task benchmark | *(running)* | *(running)* | vs GORP's −0.7 (7B) |

**It's not a math trick.** On maximally different domains (code → math → instructions → science), avr-cl achieves BWT −0.010 — near-zero forgetting. On a different architecture (LFM2.5, hybrid conv+attention), it beats both Naive and EWC.

**EWC barely helps.** On LFM2.5, EWC gets BWT −0.267 vs Naive's −0.280. The Fisher penalty slows forgetting but doesn't prevent it. avr-cl detects and repairs — fundamentally different.

## Use it

**Option 1: Full loop** — avr-cl handles everything:

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

**Option 2: As a layer** — keep your existing TRL/Axolotl training, add avr-cl between stages:

```python
import avr
from trl import SFTTrainer

trainer = SFTTrainer(model, train_dataset=task_a)
trainer.train()

snapshot = avr.get_lora_state(model)
# ... train on task B ...
drift = avr.check_drift(current_ppls, best_ppls, completed_tasks, threshold=1.15)
if drift:
    avr.repair(model, snapshot, alpha=0.1)
```

The layer API: `avr.get_lora_state()`, `avr.check_drift()`, `avr.repair()`. Plugs into any training framework.

## Try it now

A 10-minute Colab notebook that shows the repair loop firing in real-time:

**[Quickstart Colab — coming soon]**

Or reproduce the headline result:

```bash
python scripts/avr_cl_math_qwen3_1.7b.py
```

## Why not just...?

| Approach | Problem |
|---|---|
| **Replay buffers** | Need to store old training data. Privacy, plumbing, maintenance. |
| **Retrain from scratch** | Expensive. Days of compute for every new task. |
| **EWC** | Fisher penalty slows forgetting but doesn't prevent it. Barely better than Naive. |
| **mergekit** | Merges models *after* the fact. avr-cl prevents damage *during* training. Complementary. |
| **TRL / Axolotl / Unsloth** | Great training frameworks — but none check if your model forgot between stages. avr-cl plugs into them. |
| **Letta / memory layers** | Handles the retrieval layer for agents. avr-cl handles the weight layer. Use both. |

avr-cl needs zero old data, zero gradients at repair time, one LoRA snapshot, and it *knows* when the model forgot. It's not a replacement for your training framework — it's the layer that watches for forgetting between stages.

## What's next

- **PyPI**: `pip install avr-cl` (live now)
- **Quickstart Colab**: 10-minute demo on Qwen3-0.6B
- **arXiv preprint**: full method + experiments
- **DPO/GRPO support**: the 2026 post-training frontier — continual RL without forgetting
- **7B+ validation**: scaling up

**GitHub**: [github.com/ARYAN2302/tiny-cl](https://github.com/ARYAN2302/tiny-cl)
**PyPI**: [pypi.org/project/avr-cl](https://pypi.org/project/avr-cl/)
**Benchmarks**: [BENCHMARKS.md](https://github.com/ARYAN2302/tiny-cl/blob/main/BENCHMARKS.md)

---

*Fine-tune sequentially. avr-cl checks if you broke the model. And fixes it.*
