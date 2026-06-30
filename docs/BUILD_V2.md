# Tiny-CL V2: The Living Model

## Making Small Pretrained Models Continual Through Continuous Absorption

A pretrained 350M-parameter model that never stops learning — absorbing new domains one after another without forgetting, without storing any training data, using only compressed representation snapshots.

---

## What Changed From V1

V1 proved Anchor-AVR works on a 30M model trained from scratch (3x less forgetting than all baselines). V2 asks the harder, more important question:

**Can we take an existing pretrained model and make it continual?**

This is what the industry actually needs. No one trains from scratch. Everyone fine-tunes pretrained models. And every fine-tune destroys old knowledge. We fix that.

| | V1 | V2 |
|---|---|---|
| Model | 30M GPT-2 (from scratch) | SmolLM2-360M / LFM2.5-350M (pretrained) |
| Training | Train from random weights | Fine-tune with LoRA (base model frozen) |
| Domains | Stories → Wiki → News | Medical → Code → Creative Writing |
| Key mechanism | Anchor-pull on all weights | Anchor-pull on LoRA adapter weights only |
| Framing | "Continual learning" | **"Continuous Absorption"** — the model is alive |
| Target audience | CL researchers | Everyone who fine-tunes models |
| Internship angle | Generic ML | Liquid AI specifically (LFM2.5 run) |

---

## The Idea (Refined)

The model is a living system. It:

1. **Absorbs** new knowledge through LoRA fine-tuning on new domains
2. **Verifies** it hasn't broken old knowledge by checking anchor snapshots (compressed representations, NOT data)
3. **Self-repairs** when drift is detected — pulling representations back
4. **Repeats** forever — it never needs retraining from scratch

The key V2 insight: **we freeze the base model and only train LoRA adapters**. This means:
- The pretrained model's general knowledge is never destroyed (it's frozen)
- Anchor-AVR only monitors the adapter layers for drift
- Each domain gets its own adapter, but anchors ensure they don't interfere
- Storage is minimal: ~50KB anchors per domain, ~2MB per LoRA adapter

---

## Base Model Options

### Primary: SmolLM2-360M (Safe Run)

| Property | Value |
|----------|-------|
| Parameters | 360M |
| Architecture | LLaMA-style (RoPE, SwiGLU, RMSNorm) |
| Training data | ~2T tokens (curated) |
| HuggingFace ID | `HuggingFaceTB/SmolLM2-360M` |
| `output_hidden_states` | Works out of the box |
| LoRA/PEFT support | Full support (standard LLaMA layers) |
| Risk level | LOW — everything is standard |
| Why this model | HF's own flagship small model; headline "Making SmolLM2 Continual" carries HF credibility |

### Moon Shot: LFM2.5-350M (Liquid AI Run)

| Property | Value |
|----------|-------|
| Parameters | 350M |
| Architecture | Hybrid: 10 gated short conv blocks + 6 GQA attention blocks (non-transformer) |
| Training data | Proprietary (large scale) |
| HuggingFace ID | `LiquidAI/LFM2.5-350M` |
| `output_hidden_states` | UNKNOWN — needs smoke test |
| LoRA/PEFT support | UNKNOWN — custom architecture, may need manual LoRA targets |
| Risk level | HIGH — custom model code, may not expose hidden states |
| Why this model | Liquid AI's core claim is that their models are inherently adaptable. Proving Anchor-AVR works on their model validates their thesis. If they see this, internship is realistic. |

### Ablation: SmolLM2-135M

Same architecture as 360M but smaller. Used to show Anchor-AVR scales across model sizes. Ties back to V1 results.

---

## Data: Three Domains (Distinct, Real-World)

| Phase | Dataset | Domain | Tokens | Why |
|-------|---------|--------|--------|-----|
| A | `pubmed_qa` + PubMed abstracts | Medical | ~3M | Specialized vocabulary, very different from general English |
| B | `code_alpaca` or `bigcode/the-stack` (Python subset) | Code | ~3M | Structured syntax, worst case for forgetting (most different from natural language) |
| C | `artemisklnov/creative-writing` or ROCStories | Creative writing | ~3M | Natural language again but stylistically different — tests if anchors over-constrain |

**Why these domains**: Maximum distribution shift. If Anchor-AVR survives Medical → Code → Creative, it works on anything. Medical and Code are also the two domains enterprises fine-tune on most — this is not toy data.

---

## Methods

| # | Method | Mechanism | Stores Data? | Storage |
|---|--------|-----------|-------------|---------|
| 1 | Naive LoRA | Sequential adapter training, no protection | No | ~2MB/adapter |
| 2 | LoRA + Replay | Store 1% of old data, mix into training | **Yes** | ~400KB + adapters |
| 3 | LoRA + EWC | Elastic weight consolidation on adapter params | No | ~2MB/adapter (Fisher info) |
| 4 | **LoRA + Anchor-AVR (Continuous)** | Anchor-pull loss always active during adapter training | No | ~50KB anchors + adapters |
| 5 | **LoRA + Anchor-AVR (Discrete)** | Verify → targeted repair only when drifted | No | ~50KB anchors + adapters |

Method 5 is our flagship. It implements the full Absorb-Verify-Repair loop on pretrained models with LoRA.

---

## How Anchor-AVR + LoRA Works (V2)

### Setup
1. Load pretrained SmolLM2-360M, **freeze all base weights**
2. Attach LoRA adapters (rank=16, targeting q_proj and v_proj in attention layers)
3. Only LoRA parameters are trainable (~1-2M params, <1% of model)

### After Phase A (Medical):
1. Select 50 probe sequences from Phase A validation set
2. Run probes through model (with LoRA-A active), save hidden states per layer
3. These are the Phase A anchors — compressed snapshots, NOT medical data

### During Phase B (Code) — Continuous mode:
- Train LoRA-B on code data
- Every `anchor_freq` steps, also compute anchor-pull loss:
  - Run Phase A probes through model (with LoRA-B active)
  - Compare hidden states to Phase A anchors
  - `L_anchor = MSE(current_hidden, anchor_hidden)`
  - `L_total = L_lm_code + λ * L_anchor`
- This gently prevents LoRA-B from destroying Phase A representations

### During Phase B (Code) — Discrete mode (our best):
- Train LoRA-B on code data normally
- Every `verify_freq` steps:
  - Run Phase A probes through model
  - Check per-layer drift: `drift = MSE(current, anchor)`
  - If drift > threshold on any layer → PAUSE training
  - Run `repair_steps` of anchor-pull on LoRA-B weights only
  - Verify repair worked
  - Resume code training
- Only fixes what's broken, leaves everything else alone

### Why this is novel vs existing work:
- **SLIM (NAACL 2025)**: Uses MoE router + LoRA + identity layers for CL. Requires a trained router network. Complex. Only tested on 7B+ models.
- **Ours**: No router needed. Anchors detect drift automatically. Tested on 360M (the scale no one touches). Data-free. Simpler.

---

## Experiment Plan

### Step 0: Smoke Test (5 min, ~$0.15)

Before committing credits, verify the pipeline works:

```python
# Test 1: Can we load SmolLM2-360M and get hidden states?
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
outputs = model(input_ids, output_hidden_states=True)
assert outputs.hidden_states is not None

# Test 2: Can we attach LoRA?
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
peft_model = get_peft_model(model, lora_config)
assert peft_model.print_trainable_parameters() works

# Test 3: Can we compute anchor loss?
# Run probes through model, save hidden states, compute MSE
# Verify gradients flow to LoRA params only

# Test 4 (LFM2.5 only): Repeat tests 1-3 with LiquidAI/LFM2.5-350M
# This is the risky one — may fail on hidden states or LoRA targets
```

### Step 1: Sanity Run — Same-Domain Incremental (~30 min, ~$1.5)

SmolLM2-360M, general English → more general English.

Methods: Naive LoRA, Anchor-AVR Discrete only (2 methods, quick check).

Purpose: verify the pipeline works end-to-end on a pretrained model before burning credits on the hero experiment.

| # | Method | Est. Time |
|---|--------|-----------|
| 1 | Naive LoRA | ~12 min |
| 2 | LoRA + Anchor-AVR Discrete | ~15 min |

If this fails, debug before proceeding. If it works, move to Step 2.

### Step 2: Hero Run — Max-Shift Domains (~2 hours, ~$6)

SmolLM2-360M, Medical → Code → Creative Writing.

| # | Method | Est. Time |
|---|--------|-----------|
| 1 | Naive LoRA | ~20 min |
| 2 | LoRA + Replay 1% | ~25 min |
| 3 | LoRA + EWC | ~25 min |
| 4 | LoRA + Anchor-AVR Continuous | ~25 min |
| 5 | LoRA + Anchor-AVR Discrete | ~30 min |

### Step 3: LFM2.5-350M Moon Shot (IF smoke test passes, HARD CAP $8)

Same 5 methods on LFM2.5-350M. Budget capped at $8 total for ALL LFM2.5 experiments.
If any single run threatens to exceed budget (too slow, OOM), kill it and fall back to SmolLM2-only results.

### Step 4: SmolLM2-135M Ablation (~1 hour, ~$3)

Run methods 1, 4, 5 only (Naive + both Anchor-AVR variants). Purpose: show scaling across model sizes.

### Budget Summary

| Item | Cost |
|------|------|
| Smoke test | ~$0.15 |
| Sanity run (same-domain) | ~$1.50 |
| Hero run (max-shift, SmolLM2-360M) | ~$6 |
| LFM2.5-350M moon shot (hard cap) | ~$8 max |
| SmolLM2-135M ablation | ~$3 |
| Debugging / re-runs | ~$3 |
| **Total worst case** | **~$21.50** |
| **Likely actual** | **~$15** |
| **Remaining Modal credits** | **$21** |

If LFM2.5 fails smoke test, total drops to ~$13.50. Plenty of margin.

---

## Training Hyperparameters

### Base (same for all methods):

```python
learning_rate = 2e-4          # Slightly lower for fine-tuning
batch_size = 16               # Smaller batch for LoRA
context_length = 512          # Longer context (pretrained model can handle it)
epochs_per_phase = 3          # Fewer epochs (pretrained model converges faster)
weight_decay = 0.01
warmup_steps = 50
lr_scheduler = "cosine"
max_grad_norm = 1.0
seed = 42
```

### LoRA config:

```python
lora_rank = 16
lora_alpha = 32               # 2x rank (standard practice)
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]  # For LLaMA-style models
# For LFM2.5: TBD based on smoke test — may need ["attention.wq", "attention.wv"]
```

### Method-specific:

```python
# Replay
replay_buffer_pct = 0.01
replay_mix_ratio = 0.25

# EWC
ewc_lambda = 0.1
ewc_fisher_n_samples = 200

# Anchor-AVR
n_anchor_probes = 50
anchor_loss_weight = 1.0
anchor_freq = 10              # Continuous mode
verify_freq = 100             # Discrete mode
drift_threshold = 0.1
repair_steps = 50
repair_lr = 1e-4
```

---

## Project Structure (V2)

```
tiny-cl/
├── BUILD.md              ← V1 spec (archive)
├── BUILD_V2.md           ← V2 spec (this file)
├── config.py             ← V1 configs (still used by V1 code)
├── model.py              ← V1 model (archived)
├── data.py               ← V1 data pipeline (archived)
├── methods.py            ← V1 methods (archived)
├── train.py              ← V1 training (archived)
├── evaluate.py           ← V1 evaluation (archived)
├── run_all.py            ← V1 entry point (archived)
├── plot_results.py       ← V1 plotting (archived)
├── modal_run.py          ← V1 Modal wrapper (archived)
│
├── v2/                   ← ALL NEW CODE
│   ├── config.py         ← V2 configs (pretrained models, LoRA, 3 domains)
│   ├── data.py           ← V2 data pipeline (medical, code, creative)
│   ├── methods.py        ← V2 methods (LoRA + Replay, EWC, Anchor-AVR)
│   ├── anchors.py        ← Anchor store (adapted for LoRA + pretrained models)
│   ├── train.py          ← V2 training loop (LoRA fine-tuning per phase)
│   ├── evaluate.py       ← V2 evaluation (perplexity + zero-shot tasks)
│   ├── run_all.py        ← V2 experiment runner
│   ├── smoke_test.py     ← Pre-flight checks before spending credits
│   └── modal_run.py      ← V2 Modal wrapper
```

**Important**: V1 code stays untouched. V2 is a clean `v2/` directory. No modifications to existing working code.

---

## Evaluation

### Core metrics (same as V1):

After each training phase, evaluate on held-out data from ALL phases:

- **Perplexity per phase** (lower = better)
- **Forgetting Factor**: `ppl_after_drift / ppl_after_learn` (1.0 = perfect, higher = more forgetting)
- **Backward Transfer (BWT)**: average change in old-phase performance

### V2 additions:

- **Zero-shot general English**: Test on WikiText-103 to verify the pretrained base isn't degraded
- **Cross-domain transfer**: Can the medical adapter help explain medical code? Can the code adapter help structure creative writing?
- **LoRA parameter drift**: Track how much LoRA weights change between phases (signal for overfitting)

### The killer visualization:

```
Memory Health Dashboard
┌─────────────────────────────────────────────┐
│ Phase A (Medical)  ████████████████░░░░ 82% │ ← anchors pulling it back
│ Phase B (Code)     ████████████████████ 100% │ ← currently training
│ Phase C (Creative) (not yet learned)         │
│                                              │
│ General English    ████████████████████ 99%  │ ← base model untouched
└─────────────────────────────────────────────┘
```

This is what gets shared. Not a table. A dashboard showing the model is alive and healthy.

---

## What Success Looks Like

### Minimum viable result:

| Method | Phase A ppl | Phase B ppl | Phase C ppl | Forget Factor | Storage | Stores Data? |
|--------|------------|------------|------------|---------------|---------|-------------|
| Naive LoRA | 45+ | 38+ | 35 | 4-8x | ~6MB | No |
| LoRA + Replay | 30 | 28 | 27 | 2-3x | ~6MB | **Yes** |
| LoRA + EWC | 35 | 32 | 30 | 2-4x | ~6MB | No |
| **Anchor-AVR Cont** | **28** | **27** | **26** | **1.5-2x** | **~6MB** | **No** |
| **Anchor-AVR Disc** | **26** | **26** | **25** | **1.2-1.5x** | **~6MB** | **No** |

### The headline:

**"The First Living Small Model: Continuous Absorption Without Forgetting"**

Or for Liquid AI specifically:

**"Making Liquid Foundation Models Continual: Anchor-Guided LoRA Without Stored Data"**

### The one-liner:

"A 350M pretrained model that absorbs new domains one after another without forgetting — using zero stored training data, just tiny representation snapshots."

---

## Next Steps (Ordered)

### Phase 1: Smoke Test (TODAY, 15 min)

1. Create `v2/` directory and `smoke_test.py`
2. Test SmolLM2-360M loading + hidden states + LoRA attachment
3. Test LFM2.5-350M loading + hidden states + LoRA attachment
4. Document what works and what doesn't for LFM2.5
5. Decision point: proceed with LFM2.5 or fall back to SmolLM2 only

### Phase 2: Data Pipeline + Sanity Run (Day 1)

1. Write `v2/data.py` — download and tokenize all datasets
   - Sanity: general English (WikiText-2 split into 2 halves)
   - Hero: medical (PubMed), code (CodeAlpaca), creative (ROCStories)
2. Verify token counts and data quality
3. Write `v2/config.py` — model configs, LoRA configs, domain configs
4. Quick sanity run on Modal: Naive LoRA + Anchor-AVR Discrete on same-domain incremental
5. Debug any pipeline issues on the easy setting before the hero run

### Phase 3: Core Methods (Day 2)

1. Write `v2/anchors.py` — anchor store adapted for LoRA + frozen base model
   - Hidden-state anchors as primary (consistent with V1)
   - Weight-delta anchor as ablation flag (for later)
2. Write `v2/methods.py` — 5 methods (Naive LoRA, Replay, EWC, Anchor-AVR Cont, Anchor-AVR Disc)
3. Write `v2/train.py` — training loop with LoRA fine-tuning per phase
4. Write `v2/evaluate.py` — perplexity, forgetting factor, BWT

### Phase 4: Hero Run — SmolLM2-360M Max-Shift (Day 3)

1. Write `v2/modal_run.py` for cloud execution
2. Run all 5 methods on SmolLM2-360M with Medical → Code → Creative
3. This is the core result — everything else is bonus

### Phase 5: LFM2.5-350M Moon Shot (Day 3-4, IF smoke test passed, HARD CAP $8)

1. Adapt LoRA target modules based on LFM2.5's architecture
2. Run all 5 methods
3. Kill any run that threatens to exceed $8 total for LFM2.5
4. This is the internship play — if results are good, this goes to Liquid AI

### Phase 6: SmolLM2-135M Ablation (Day 4)

1. Naive + both Anchor-AVR variants only
2. Shows scaling behavior
3. Optional: run weight-delta anchor ablation here (1 method, 1 phase)

### Phase 7: Results & Framing (Day 5)

1. Generate forgetting curves, comparison tables
2. Build the "Memory Health Dashboard" visualization
3. Dual framing:
   - Paper/abstract: "Anchor-based continual adaptation for pretrained small models"
   - Blog/README/hero: "The Living Model: continuous absorption without forgetting"
4. Include sanity run as "we also validated on incremental same-domain" paragraph

### Phase 8: Ship It (Day 6)

1. Clean GitHub repo (V1 + V2 results)
2. Write 4-page research note with dual framing
3. Post results — tweet the dashboard, not the table
4. Email Liquid AI research team with LFM2.5 results (if available)
5. Apply to internships

---

## Design Decisions (Resolved)

### 1. LFM2.5-350M: Bonus, Not Dependency

Primary success path is SmolLM2-360M. LFM2.5-350M is explicitly a "moonshot add-on."
- If smoke test fails or it's unstable/slow, skip it entirely
- The paper and story are already strong with SmolLM2 + LoRA + anchors
- LFM2.5 is "bonus figures + internship angle," not core infrastructure

### 2. Anchor Design: Hidden States Primary, Weight-Delta Ablation

Hidden-state anchors are the main method. Weight-delta anchoring is a small side experiment.

Why hidden states first:
- They preserve functional behavior on probe inputs, not numerical weights
- Consistent with V1 story: "model monitors its own representations and self-corrects"
- LoRA weight deltas can drift in ways that still preserve behavior; anchoring them directly risks over-constraining and killing plasticity

Weight-delta ablation (run for one phase/size only):
- If it's worse (likely): extra result — "representation-level anchoring beats direct parameter anchoring"
- If it's better: we rethink, but it's unlikely

### 3. Framing: Dual-Layer Branding

- **Technical core (paper/abstract)**: "Anchor-based continual adaptation for pretrained small models"
- **Story/slogan (blog/README/hero image)**: "The Living Model: continuous absorption without forgetting"
- "Continuous Absorption" is a section concept, not the only top-level brand
- Researchers latch onto "anchor-based continual learning"; broader audience likes "living model"

### 4. Domains: Safe Run First, Then Hero Run

**Phase 0 (sanity check)**: Same-domain incremental
- General English → more general English (or medical → more medical)
- Easier to stabilize, good place to debug AVR + LoRA
- Still meaningful: showing continual pre-training without replay
- Not the headline result; used for debugging + a "we also checked incremental" paragraph

**Phase 1 (hero experiment)**: Max-shift sequence
- Medical → Code → Creative Writing
- If it works here, it's clearly not a toy
- If it only partially works, that's still interesting and reportable

### 5. Budget: Hard-Cap LFM2.5 at $8

- Smoke test for LFM2.5 first (cents)
- If it passes, allocate at most $6–8 total for ALL LFM2.5 experiments combined
- If any single run threatens to exceed that (too slow, OOM), kill it early and fall back to SmolLM2-only results
- Core SmolLM2-360M + 135M runs + re-runs get the majority of budget

### 6. Baselines

- Naive LoRA (no protection)
- LoRA + Replay 1% (data-dependent)
- LoRA + EWC (data-free, same no-storage constraint)
- LoRA + Anchor-AVR Continuous (ours)
- LoRA + Anchor-AVR Discrete (ours, flagship)

EWC is the strongest fair baseline — same storage, same no-data constraint.
O-LoRA is too complex to implement correctly in this timeline; skip it.

### 7. Evaluation

Primary: perplexity per phase, forgetting factor, BWT.
Secondary (if compute allows): zero-shot task accuracy on MedQA / HumanEval.
Do NOT sacrifice core perplexity results for task accuracy — it's a nice-to-have.

---

## V1 Results (Reference)

These are the results that justify V2:

| Method | Phase A ppl | Forgetting Factor | Storage |
|--------|------------|-------------------|---------|
| Naive SGD | 242 | 8.3x | 0 KB |
| Freeze | 143 | 4.9x | 0 KB |
| Replay 1% | 292 | 10.1x | 824 KB |
| **AVR Continuous** | **92** | **3.2x** | **1725 KB** |
| **AVR Discrete** | **86** | **3.0x** | **1725 KB** |

Key V1 findings:
- Anchor-AVR gives 3x less forgetting than all baselines
- Discrete mode (verify-then-repair) beats continuous mode
- Replay is COUNTERPRODUCTIVE at sub-50M scale (10.1x vs 8.3x naive)
- Freezing helps but limits plasticity (highest Phase A perplexity among non-naive methods)

V2 tests whether these findings hold on pretrained models at 10x the scale.
