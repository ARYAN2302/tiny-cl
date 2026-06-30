# Tiny-CL: Continual Learning at Sub-50M Scale

## Learning Without Storing: Self-Correcting Continual Language Learning

A 30M parameter language model that keeps learning after deployment — without storing any old training data.

---

## The Idea

Instead of storing old data and replaying it (standard approach), the model:
1. **Absorbs** new knowledge normally
2. **Verifies** whether old knowledge has degraded by checking its own internal representations against compressed anchor snapshots
3. **Self-corrects** by pulling drifted representations back toward the anchors
4. **Repeats** — keeps learning, checking, and self-correcting forever

No replay buffer. No stored user data. Just tiny anchor snapshots (~50KB) and self-correction.

---

## Project Structure

```
tiny-cl/
├── BUILD.md           ← You are here
├── config.py          ← Model sizes, training params, method configs
├── model.py           ← Tiny GPT-2 model (15M / 30M params)
├── data.py            ← Download + tokenize + split data into 3 phases
├── methods.py         ← CL methods: Naive, Freeze, Replay, AnchorAVR
├── train.py           ← Main training loop (single experiment)
├── evaluate.py        ← Perplexity, BWT, FWT metrics
├── run_all.py         ← Run all experiments, save results
├── plot_results.py    ← Generate forgetting curves + comparison plots
├── requirements.txt   ← pip dependencies
└── modal_run.py       ← Modal wrapper for cloud execution
```

---

## Architecture Details

### Model: Tiny GPT-2

| Config | n_layer | n_head | n_embd | n_ff | Vocab | Total Params |
|--------|---------|--------|--------|------|-------|-------------|
| 30M    | 6       | 8      | 384    | 1536 | 50257 | ~30.3M      |
| 15M    | 4       | 4      | 256    | 1024 | 50257 | ~16.3M      |

Uses GPT-2 tokenizer with weight tying (embedding = output projection).

### Data: Three Phases (10M words total)

| Phase | Dataset | Domain | Words | ~Tokens |
|-------|---------|--------|-------|---------|
| A     | TinyStories | Children's stories, simple syntax | 4M | ~5.3M |
| B     | WikiText-2 | Wikipedia-style expository text | 3M | ~4M |
| C     | AG News | News headlines/descriptions, diverse topics | 3M | ~4M |

Each phase is seen once (single epoch) for the continual learning setup.
Additional epochs allowed for ablations.

### Methods

| Method | Mechanism | Stores Data? | Storage Overhead |
|--------|-----------|-------------|------------------|
| Naive SGD | Sequential training, no protection | No | 0 |
| Freeze Layers | Freeze bottom N layers after Phase A | No | 0 |
| Blind Replay 1% | Store 1% of old data, mix into training | Yes | ~400KB |
| **Anchor-AVR (continuous)** | Anchor-pull loss always active | No | ~50KB (anchors only) |
| **Anchor-AVR (discrete)** | Verify → targeted repair only when drifted | No | ~50KB (anchors only) |

---

## How Anchor-AVR Works

### After training on Phase A:
1. Select 200 probe sequences from Phase A validation set
2. Run them through the model and save hidden states at each layer
3. These compressed snapshots are the "anchors" — not the data, just the model's state

### During training on Phase B:

**Continuous mode:**
- Every training step, also compute anchor-pull loss
- `L_total = L_lm + λ * L_anchor`
- `L_anchor = Σ MSE(current_hidden[layer], anchor_hidden[layer])`
- This gently keeps representations stable while learning new things

**Discrete mode (your original idea):**
- Train on Phase B normally
- Every N steps, verify: run probes, check drift per layer
- If drift > threshold on any layer: pause, do targeted anchor-pull repair
- Only fix what's broken, leave everything else alone

### Key difference from replay:
- Replay stores and re-shows old DATA
- Anchor-AVR stores old REPRESENTATIONS (tiny, no privacy issues)
- Replay needs the original text
- Anchor-AVR only needs compressed hidden state snapshots

---

## Experiment Plan

### Required runs (do these first):

| # | Method | Model | Time Est. |
|---|--------|-------|-----------|
| 1 | Naive SGD | 30M | ~7 min |
| 2 | Freeze layers (bottom 3) | 30M | ~7 min |
| 3 | Blind Replay 1% | 30M | ~10 min |
| 4 | Anchor-AVR continuous | 30M | ~12 min |
| 5 | Anchor-AVR discrete | 30M | ~15 min |

### Nice-to-have (if budget allows):

| # | Method | Model | Time Est. |
|---|--------|-------|-----------|
| 6 | Naive SGD | 15M | ~4 min |
| 7 | Blind Replay 1% | 15M | ~6 min |
| 8 | Anchor-AVR continuous | 15M | ~8 min |

**Total required: ~51 minutes of GPU time (~$1.30 on A100)**
**Total with nice-to-have: ~69 minutes (~$1.75)**

Plenty of budget left for debugging and re-runs.

---

## Training Hyperparameters

```python
learning_rate = 3e-4
batch_size = 32
context_length = 256
epochs_per_phase = 5
weight_decay = 0.1
warmup_steps = 100
lr_scheduler = "cosine"
```

### Method-specific:

```python
# Freeze
n_freeze_layers = 3  # out of 6

# Replay
replay_buffer_pct = 0.01  # 1% of old data
replay_mix_ratio = 0.25   # 25% of each batch is replay data

# Anchor-AVR
n_anchor_probes = 200
anchor_loss_weight = 1.0
verify_freq = 100         # verify every 100 steps
drift_threshold = 0.1     # MSE threshold for discrete mode
repair_steps = 50         # steps of targeted repair
repair_lr = 1e-4
```

---

## Evaluation

After each training phase, evaluate on held-out data from ALL phases:

**Metrics:**
- **Perplexity** per phase (lower = better)
- **Backward Transfer (BWT):** average change in old-phase performance after learning new phase
  - BWT < 0 means forgetting
  - BWT > 0 means backward improvement (rare but possible)
- **Forward Transfer (FWT):** zero-shot performance on future phases (optional)

**The forgetting curve:**
```
Perplexity on Phase A
  |
  |  After A: 45    After B: 68    After C: 92    ← Naive (forgot everything)
  |  After A: 45    After B: 48    After C: 51    ← Anchor-AVR (barely forgot)
  +-----------------------------------------------
```

This is the key visualization. The table is for the paper. The plot is for the tweet.

---

## Results Format

Results saved to `results/` directory:
- `results_<method>_<model_size>.json` — raw metrics per phase
- `summary.csv` — all methods compared
- `forgetting_curves.png` — the key plot
- `comparison_table.png` — bar chart comparison

---

## How to Run

### Local (with GPU):
```bash
pip install -r requirements.txt
python run_all.py --model-size 30M --methods naive freeze replay anchor_cont anchor_disc
```

### On Modal:
```bash
pip install modal
python modal_run.py
```

### Quick test (CPU, tiny data):
```bash
python run_all.py --model-size 15M --methods naive anchor_cont --debug
```

---

## What Success Looks Like

The minimum viable result:

| Method | Phase A ppl | Phase B ppl | Phase C ppl | BWT | Storage |
|--------|------------|------------|------------|-----|---------|
| Naive SGD | 90+ | 65+ | 45 | -40 | 0 |
| Freeze | 55 | 75+ | 70+ | -15 | 0 |
| Replay 1% | 48 | 50 | 44 | -6 | 400KB |
| **Anchor-AVR** | **47** | **49** | **44** | **-3** | **50KB** |

*(Numbers are aspirational, not guaranteed)*

**The headline:** Anchor-AVR matches replay with 8x less storage and zero stored training data.

**The one-liner:** "A tiny language model that keeps learning after you deploy it, and it doesn't need to store your data to remember."

---

## Timeline

| Day | Task |
|-----|------|
| 1-2 | Set up pipeline, download data, verify model trains on Phase A |
| 3 | Run Naive SGD + Freeze (baselines) |
| 4-5 | Run Replay + Anchor-AVR methods |
| 6 | Debug, re-run if needed, bonus runs |
| 7 | Generate plots, clean results |
| 8 | Write 4-page note, clean GitHub repo |
| 9 | Post, email researchers, apply |
