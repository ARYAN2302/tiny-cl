# avr-cl

**Your fine-tune silently broke your model. avr-cl checks if it broke, and fixes it.**

A continual post-training framework with three phases: LEARN → VERIFY → REPAIR. After each fine-tuning stage, it detects if the model forgot prior tasks and repairs the damage in weight space — no replay buffer, no old training data, one LoRA snapshot in memory.

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

### Headline — Qwen3-1.7B (5000 examples/task)

4-task math stream: GSM8K → MATH → AQuA-RAT → SVAMP. LoRA r=128, 3 epochs.

| | Naive SFT | EWC | **avr-cl** |
|---|---|---|---|
| **BWT** | −0.453 | — | **−0.078** |
| **GSM8K after all 4 tasks** | 9% | — | **47%** |
| **ACC** | 0.220 | — | **0.529** |
| **Repair steps** | 0 | 0 | **29** |

**5.8× less forgetting.** The repair loop fired 29 times across 3 task transitions.

### Cross-model validation — 500 examples/task

Same 4-task math stream, reduced data for fast iteration.

| Model | Method | BWT | ACC | Repairs |
|---|---|---|---|---|
| Qwen3-1.7B | Naive | −0.320 | 0.240 | 0 |
| Qwen3-1.7B | **AVR** | **−0.037** | **0.522** | **27** |
| LFM2.5-1.2B | Naive | −0.280 | 0.198 | 0 |
| LFM2.5-1.2B | EWC | −0.267 | 0.232 | 0 |
| LFM2.5-1.2B | **AVR** | **−0.150** | **0.357** | **30** |

**AVR beats both Naive and EWC on every model.** EWC barely outperforms Naive — the Fisher penalty slows forgetting but doesn't prevent it. AVR detects and repairs it.

### Cross-domain — Qwen3-1.7B

4 maximally unrelated domains: Code → Math → Instruct → Science.

| Method | BWT | ACC | Repairs |
|---|---|---|---|
| **AVR** | **−0.010** | **0.667** | **17** |

Near-zero forgetting across maximally different domains. Not just a math trick.

### TRACE 8-task benchmark

The standard CL-LLM benchmark. 8 diverse tasks: C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten.

| Method | BWT | ACC | Repairs |
|---|---|---|---|
| Naive | *(running)* | | |
| **AVR** | *(running)* | | |

Published baselines on TRACE 8-task (7B models): GORP (ACL 2025) BWT = −0.7, O-LoRA BWT = −4.3, CoDyRA BWT = −3.25.

Full results: [`BENCHMARKS.md`](BENCHMARKS.md) · [`results/`](results/)

## Install

```bash
pip install avr-cl
```

*Until PyPI release: `pip install git+https://github.com/ARYAN2302/tiny-cl.git`*

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

**Custom repair operator** (TIES, TaskArithmetic, etc.):

```python
import avr

def my_repair(model, snapshot, alpha, device):
    # your merge logic here
    return n_params_touched

result = avr.run(model=..., tasks=..., repair_fn=my_repair)
```

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
| **mergekit** | Merges N separately-trained models *after* the fact. avr-cl prevents the damage *during* one training run. |
| **Just use TRL** | TRL has no concept of "after this task, before the next." It doesn't know your model forgot. |
| **Letta / memory layers** | Handles the retrieval layer. avr-cl handles the weight layer. Use both. |

avr-cl needs zero old data, zero gradients at repair time, one LoRA snapshot, and it *knows* when the model forgot.

## Reproduce

```bash
# Headline result (Qwen3-1.7B, 5000 examples/task, ~5h on Kaggle T4)
python scripts/avr_cl_math_qwen3_1.7b.py

# Baseline comparison (Qwen3-1.7B, 500 examples/task, ~3h)
python scripts/exp5_baselines_qwen3.py

# Baseline comparison (LFM2.5-1.2B, 500 examples/task, ~2.5h)
python scripts/exp6_baselines_lfm.py

# TRACE 8-task benchmark (~5h)
python scripts/exp3_trace_8task.py
```

All experiment scripts are **self-contained** — paste into a Kaggle notebook with GPU T4 enabled and run. No external setup needed.

## Limitations

- **Validated on 1.7B and 1.2B.** 7B+ is on the roadmap.
- **SFT only.** DPO/GRPO support is the next milestone — the 2026 post-training frontier.
- **Repair cap.** `max_repair_steps=10` sometimes hits before full convergence on hard transitions. Bumping to 20 may help on difficult task streams.

## Roadmap

- [ ] PyPI release (`pip install avr-cl`)
- [ ] Quickstart Colab notebook
- [ ] HuggingFace Hub integration (`avr.push_to_hub`)
- [ ] DPO/GRPO support for the LEARN phase
- [ ] 7B+ model validation
- [ ] arXiv preprint

## License

MIT.
