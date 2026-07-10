# Post-Training Comparison Plan — avr-cl vs the field

## The goal

AVR is validated on TRACE (continual learning benchmark). Now we prove it works in a **post-training pipeline** — the actual use case the framework ships for. We compare AVR against:

1. **The base model** (no post-training) — the floor
2. **Naive sequential SFT** (what Llama-Factory / Unsloth do if you fine-tune sequentially) — the baseline
3. **Other PT frameworks** (Llama-Factory, Unsloth) — the competition

AVR must beat all three on the question that matters: **can you post-train on a stream of tasks/domains without the model degrading?**

---

## What "post-training" means here

Not single-task fine-tuning. A **stream**:

```
Stage 1: SFT on UltraChat (general chat capability)
Stage 2: Domain fine-tune on Medical
Stage 3: Domain fine-tune on Code
Stage 4: Domain fine-tune on Legal
```

After all 4 stages, evaluate:
- **MT-Bench** (chat quality — did Stage 1 survive?)
- **Medical accuracy** (did Stage 2 survive Stages 3-4?)
- **Code accuracy** (did Stage 3 survive Stage 4?)
- **Legal accuracy** (did Stage 4 land?)

**The base model** gets MT-Bench only (it has no domains).
**Naive SFT** does all 4 stages sequentially, no CL.
**AVR** does all 4 stages with verify-repair between each.
**Llama-Factory/Unsloth** do all 4 stages sequentially (they have no CL — same as naive, but through their tooling).

---

## The metrics

| Metric | What it measures | Who it favors |
|--------|-----------------|---------------|
| MT-Bench (after all stages) | Did general chat survive domain fine-tuning? | AVR (if it works) |
| Domain accuracy (per domain, final) | Did each domain survive later domains? | AVR |
| BWT across stages | Average backward transfer | AVR |
| ACC across all evals | Overall capability | The best method |
| Wall-clock time | How long does the pipeline take? | Naive (fastest) |
| Memory | How much state does the method need? | AVR (O(1), no replay) |

---

## The comparison matrix

| Method | Tool | CL? | Memory | What we expect |
|--------|------|-----|--------|----------------|
| Base model | — | — | O(0) | MT-Bench only, no domains |
| Naive SFT | raw transformers | ❌ | O(1) | Domains degrade each other, MT-Bench collapses |
| Llama-Factory | llamafactory | ❌ | O(1) | Same as naive (no CL), but through their UI |
| Unsloth | unsloth | ❌ | O(1) | Same as naive (no CL), faster training |
| **AVR** | **avr-cl** | ✅ | O(1) | Domains survive, MT-Bench survives |

**The key claim:** Llama-Factory and Unsloth are great at single-stage fine-tuning. But if you use them for sequential post-training, your model degrades. AVR-cl is the only framework that prevents this.

---

## The experiments

### Experiment 1: AVR on the post-training pipeline
**Config:** `avr/configs/posttraining_pipeline.yaml` (to be written)
- Model: LFM2.5-350M
- Stage 1: SFT on UltraChat-200k (10k samples, 1 epoch)
- Stage 2: Medical fine-tune (epfl-llm/guidelines, 1k samples, 2 epochs)
- Stage 3: Code fine-tune (iamtarun/python_code_instructions, 1k samples, 2 epochs)
- Stage 4: Legal fine-tune (nguha/legalbench, 1k samples, 2 epochs)
- AVR verify-repair between each stage (threshold 1.15, α=0.1, max_steps=10)
- Eval after Stage 4: MT-Bench + per-domain accuracy

**Runtime:** ~4h on Kaggle T4

### Experiment 2: Naive SFT on the same pipeline
**Same as Experiment 1 but with AVR disabled.** This is the "what happens if you use Llama-Factory for this" baseline.

**Expected:** MT-Bench collapses after Stage 2, domains degrade each other.

### Experiment 3: Base model
**Just evaluate the base LFM2.5-350M on MT-Bench and the domain test sets.** No fine-tuning. This is the floor.

**Runtime:** 30 min

### Experiment 4 (optional): Llama-Factory on the same pipeline
**Run the same 4-stage pipeline through Llama-Factory.** Proves that the tool everyone uses produces the same degradation as naive SFT.

**Runtime:** ~4h. **Lower priority** — if Llama-Factory = naive SFT (which it should, since it has no CL), this just confirms the point. Run only if we need the "we beat the actual tool" headline.

---

## The deliverable

A table for the README:

```
| Method          | MT-Bench | Medical | Code  | Legal | BWT   | Memory |
|-----------------|----------|---------|-------|-------|-------|--------|
| Base model      | 4.2      | 0.31    | 0.25  | 0.28  | —     | O(0)   |
| Naive SFT       | 2.8      | 0.22    | 0.34  | 0.35  | -0.12 | O(1)   |
| Llama-Factory   | 2.9      | 0.23    | 0.33  | 0.34  | -0.11 | O(1)   |
| **AVR-cl**      | **4.0**  | **0.30**| **0.32** | **0.33** | **-0.03** | O(1) |
```

(The numbers are illustrative — actual numbers from the runs.)

**The headline:** "AVR-cl is the only post-training framework that preserves chat quality through a domain stream. Llama-Factory and Unsloth lose 30%+ of chat capability after 3 domain fine-tunes. AVR-cl loses <5%."

---

## What needs to be built

1. **`avr/configs/posttraining_pipeline.yaml`** — the 4-stage config
2. **`avr/data.py`** — add UltraChat, Medical, Code, Legal loaders (some already exist in realworld_stream)
3. **`avr/eval.py`** (new) — MT-Bench evaluation harness (uses GPT-4 as judge, or self-eval if no API key)
4. **`experiments/posttraining_avr.py`** — the AVR pipeline run
5. **`experiments/posttraining_naive.py`** — the naive baseline run
6. **`experiments/posttraining_base.py`** — base model eval only

---

## The order

1. **Wait for v23 seed 123 (10-step cap) to finish.** This confirms the method is robust. If BWT is bad, we have a problem. If BWT is ~−0.02, we're good.
2. **Build the post-training pipeline scripts.** ~1 day, no GPU.
3. **Run Experiment 3 (base model).** 30 min. Gets us the floor.
4. **Run Experiment 1 (AVR pipeline).** 4h. The main result.
5. **Run Experiment 2 (naive pipeline).** 4h. The comparison.
6. **Fill the table.** Ship.

**Total: ~1.5 days + 8h GPU.**

---

## The honest risk

The post-training pipeline is harder than TRACE. TRACE tasks are small and similar (all classification). The post-training pipeline has:
- UltraChat (long conversations, different format)
- Medical (long text, technical)
- Code (structured, Python)
- Legal (formal, technical)

The domains are more different from each other than TRACE tasks are. **This means more drift, which means AVR has to work harder.** If the 10-step cap can't handle the drift in this pipeline, we'll see it immediately.

**Mitigation:** if AVR struggles, we can increase max_steps to 15-20 for the post-training pipeline (the cap is per-task, not global). But we start with 10 and see what happens.
