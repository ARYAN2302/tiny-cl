# Post-Training Framework Gap Research — avr-cl

Researched July 2026. Every claim below is sourced. Local context: your `avr/` package
(framework.py / operators.py / detectors.py / trainer.py / metrics.py / strategy.py / cli.py)
and your `AVR_SECOND_OPINION.md` / `POSTTRAINING_PLAN.md` were read first, so this builds
on — not duplicates — your prior work.

---

## A. Survey of post-training frameworks (July 2026)

I checked each for five CL capabilities: **(a)** continual learning across task streams,
**(b)** drift detection as a training-time signal, **(c)** weight-space repair between
stages, **(d)** backward-transfer (BWT/FF/R-matrix) as first-class metrics,
**(e)** a snapshot→verify→repair loop.

| Framework | What it actually is | a | b | c | d | e | Source |
|---|---|---|---|---|---|---|---|
| **TRL** (HF) | Single-stage post-trainers: `SFTTrainer`, `GRPOTrainer`, `DPOTrainer`, `RewardTrainer`. v1 shipped Mar 2026. New: **Harbor** = sandboxed agent task suites via `GRPOTrainer.environment_factory`. Each trainer is "a light wrapper around `transformers.Trainer`". | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/huggingface/trl ; huggingface.co/blog/trl-v1 |
| **Unsloth** | Fast kernels + Unsloth Studio (no-code UI). Full/4bit/16bit/FP8, MoE 12× faster. Scope = speed + UX, **single stage**. No CL. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/unslothai/unsloth ; unslothai.substack.com |
| **Axolotl** | YAML-config-driven fine-tuning *pipeline* (preprocess→train→eval→quant). Single config = one run. No stream notion. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/axolotl-ai-cloud/axolotl |
| **LLaMA-Factory** | Unified FT of 100+ LLMs: "(continuous) pre-training, SFT, reward modeling, PPO, DPO, KTO, ORPO". **Note: "continuous pre-training" = CPT (more raw data), NOT CL across tasks.** No drift/repair/BWT. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/hiyouga/LlamaFactory ; arxiv 2403.13372 |
| **verl** (Volcano) | Flexible/production RL library (HybridFlow). Has `docs/start/agentic_rl.rst` (backend-agent RL). Single objective. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/verl-project/verl |
| **OpenRLHF** | Ray + vLLM; Actor/Reward/Reference/Critic split across GPUs. RLHF/PPO. Distributed-first. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/openrlhf/openrlhf ; arxiv 2405.11143 |
| **SkyRL / SkyRL-Agent** | Modular long-horizon agent RL (SWE-Bench-style), built *on top of verl + OpenHands*. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/novasky-ai/skyrl ; arxiv 2511.16108 |
| **prime-rl** (Prime Intellect) | Async RL at 1000+ GPU scale (Orchestrator–Trainer), powers INTELLECT-3. Pairs with `verifiers` + `renderers`. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/PrimeIntellect-ai/prime-rl ; openreview yk3ICpEbv8 |
| **AReaL** (Alibaba) | Fully async RL; decouples generation from training; staleness-enhanced PPO. NeurIPS 2025. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/areal-project/AReaL ; arxiv 2505.24298 |
| **OpenPipe ART** | Agent RL trainer: GRPO + RULER auto-reward harness. Multi-step agents. | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/openpipe/art ; art.openpipe.ai |
| **rLLM** (agentica) | "Post-training language agents via RL." Used to train **DeepSWE** (Together AI). RL-agent, single objective. (Not "tutorial-grade" — it's Together's actual ship vehicle.) | ✗ | ✗ | ✗ | ✗ | ✗ | github.com/agentica-project/rllm ; together.ai/blog/deepswe |
| **Slime** | Smaller RL tutorial/research repo. | ✗ | ✗ | ✗ | ✗ | ✗ | (you listed it; tutorial-grade) |
| **ContinualLM** (UIC-Liu-Lab) | **The only thing with "continual" in the name for LMs.** See §B. | ◐ | ✗ | ✗ | ◐ | ✗ | github.com/UIC-Liu-Lab/ContinualLM |

**Verdict on the matrix:** zero production post-training framework ships (a)–(e). The RL
frameworks (verl/OpenRLHF/SkyRL/prime-rl/AReaL/ART/rLLM) optimize *one* reward on *one*
task distribution. The SFT frameworks (TRL/Unsloth/Axolotl/LLaMA-Factory) do *one* stage.
None model time as a stream.

---

## B. The continual-learning gap — is it real?

**Yes.** Direct verification:

1. **No `StreamTrainer` / `ContinualTrainer` class exists** in any LLM post-training
   framework. A targeted search returns only TRL's SFT/GRPO/DPO/Reward trainers and a
   HuggingFace post-training internship take-home — nothing stream-oriented.
   (github.com/huggingface/trl ; github.com/huggingface/post-training-takehome)

2. **TRL's trainers are explicitly single-dataset, single-stage.** README: "Each trainer
   in TRL is a light wrapper around the 🤗 Transformers trainer." No stream, no verify,
   no repair, no BWT. Confirmed by reading the v1 README.

3. **The closest name-collision, ContinualLM, does NOT fill the gap.** I read its README:
   - Built on `transformers==4.17.0` + `adapter-transformers==3.0.1` (2022-era).
   - Paradigm is **domain-adaptive PRE-training** (MLM on domain corpora, BERT-style), not
     post-training (SFT/DPO/RLHF) of decoder LMs.
   - Methods = DAS/CPT/DGA/CTR + EWC/DER++/HAT + NCL/ONE baselines — each a *research
     method*, trained via shell loops over `pt_task`. No pluggable LEARN/VERIFY/REPAIR.
   - No PPL-ratio drift detection firing repair *at training time*; eval is a separate
     "end-task fine-tuning" pass after all domains.
   - Not pip-installable, not LoRA/PEFT-first, not modern-HF-ecosystem.
   So ContinualLM is a 2021–2023 academic methods repo for encoder-LM domain pretraining.
   It validates the problem exists; it does not occupy the modern-decoder-post-training
   framework slot. (github.com/UIC-Liu-Lab/ContinualLM)

4. **The single-method research repos are not frameworks.** CURLoRA, CODYRA, SEE, STABLE,
   InsCL, RCL, SLAO+MVA, Group-Merger — each is one method, one repo, one paper. None
   package an abstraction. (github.com/mnoorfawi/curlora ; github.com/jeff024/codyra ;
   aclanthology.org/2025.findings-acl.387 ; arxiv 2510.16089 ; aclanthology.org/2026.mellm-1.30)

5. **The pain is recognized at every level — and nobody has shipped the framework:**
   - Academic: arxiv **2510.17776** "Mapping Post-Training Forgetting in Language Models
     at Scale" (empirically maps forgetting across the post-training pipeline).
   - Industry lab: Prime Intellect blog "Post-Training Nemotron 3 on Lab" — *"you need a
     repeatable post-training loop"* (primeintellect.ai/blog/nemotron-3). They built the
     *infra* (prime-rl/verifiers/renderers) but not the *continual* layer.
   - Serving co: Baseten "Continual learning and the post monolith AI era" — *"sequential
     SFT on seven tasks produced significant forgetting"* (baseten.co/research/continual-learning).
   - VC: a16z "Why we need continual learning" (a16z.com/why-we-need-continual-learning).
   - Benchmark: TRACE (arxiv 2310.06762, BeyonderXX/TRACE) + CoIN make the pain measurable.

**Conclusion:** the gap is real and is exactly the slot avr-cl targets — a pluggable
LEARN→VERIFY→REPAIR layer that wraps (not duplicates) TRL/Unsloth/LEAP and adds the stream,
drift detection, weight-space repair, and BWT/R-matrix they all lack.

> One honest caveat: model-merging tooling (mergekit: TIES/DARE/task-arithmetic) overlaps
> with your REPAIR operator conceptually. But mergekit is *offline batch merging of
> independently-trained models*, not a *training-time drift-triggered verify-repair loop
> inside a stream*. Different problem. Your `SnapshotInterp` could even delegate to a
> mergekit-style operator as one REPAIR impl. (github.com/arcee-ai/mergekit)

---

## C. What Liquid AI / Prime Intellect / Together actually have — and lack

### LiquidAI — **the best fit.** Their post-training story is thin and on-device CL is their problem.
- Open repos: `Liquid4All/leap-finetune` ("a minimal fine-tuning repo for LFM2") and
  `Liquid4All/cookbook`. **No CL, no stream, no drift, no repair, no BWT** — I grepped the
  LEAP README: zero hits for continual/forgetting/sequential/drift/BWT/repair/verify.
  (github.com/Liquid4All/leap-finetune)
- LEAP supports SFT/DPO/GRPO/VLM/MoE, Ray+Accelerate, YAML `extends:` config — but each
  config = one stage. Sequencing stages = silent forgetting.
- **Why they'd care:** LFM2.5-350M/1.2B are *small* (low capacity ⇒ forgets harder), and
  Liquid's on-device/per-device deployment story means models get fine-tuned sequentially
  per user/domain. That is literally a continual-learning problem. Your repo already runs
  on LFM2.5-350M + LoRA. You're demoing on *their* model, filling *their* gap.
- distillabs already fine-tunes LFM2.5-350M to 96–98% tool-calling — proving the
  fine-tune-then-deploy loop is active. (distillabs.ai/blog/fine-tuning-liquids-lfm25)

### PrimeIntellect — strong infra, no CL layer.
- Stack: `prime-rl` (async RL), `verifiers` (RL envs+evals), `renderers` (token templating),
  Hosted Training. (github.com/PrimeIntellect-ai/{prime-rl,verifiers,renderers})
- Their own words: *"you need a repeatable post-training loop"* (nemotron-3 blog). They're
  productizing repeatable post-training — but prime-rl is RL-only, single-objective. A CL
  layer that plugs into their "loop" is the obvious missing piece.
- Fit: your `Oracle`/`Consolidator` stubs rhyme with their `verifiers` philosophy.

### Together.ai — RL-agent shop, no CL.
- DeepSWE trained with `rLLM` (agentica) on R2E-Gym. Everything RL-agent, single-objective.
  (together.ai/blog/deepswe ; github.com/agentica-project/rllm)
- Nothing open on sequential/safety-preserving post-training. Gap is open but their focus
  (SWE agents) is narrower than Liquid's.

### Fastino/Modal/Replicate
- No public post-training *framework* gap statements found. Modal is compute; Replicate is
  serving. Not primary targets.

**Ranking for outreach:** Liquid AI ≫ Prime Intellect > Together. Liquid is the highest-EV
target because (1) you already run on their model, (2) their gap is the most concrete,
(3) they're a smaller lab where a DM is more likely to land.

---

## D. API design — what a `StreamTrainer` looks like

### How TRL structures trainers (confirmed from README)
- `SFTTrainer`/`GRPOTrainer`/`DPOTrainer`/`RewardTrainer` each **subclass/wrap**
  `transformers.Trainer` — "a light wrapper around the 🤗 Transformers trainer," natively
  DDP/DeepSpeed/FSDP, PEFT/LoRA integrated. Each takes **one** `train_dataset`.
- v1 added Harbor (agent sandboxes) and `trl.experimental` for unstable features.
- **There is no stream concept.** No `datasets: List[Dataset]`, no between-stage hook.

### What your code already does (and it's the right shape)
`avr/framework.py::ContinualPostTrainer` **wraps** (does not subclass) a `LearnStrategy`
with `.train(model, task, tokenizer)`. `AVRStrategy` builds detector+operator+learn from a
YAML `stream:` config and runs `run_stream(model, tokenizer, tasks, on_task_complete)`.
- Stream = ordered `List[TaskSpec]` from YAML. ✅ correct.
- Drift detection = `DriftDetector.check()` called **between tasks** (custom loop, not a
  callback). ✅ correct.
- Repair = `RepairOperator.repair_step()` in a verify-repair loop **after each task**, not
  per-epoch. ✅ correct (matches TRACE/v23).
- LoRA = `get_lora_state`/`set_lora_state` snapshot LoRA params; `SFTStrategy` sets only
  `lora_*` trainable. ✅ correct.

### Reference patterns from vision-CL libraries
- **Avalanche** (avalanche.continualai.org): `BaseStrategy` + **Plugin/Callback hooks** at
  many lifecycle points (`before_training_exp`, `after_eval_exp`…), `EvaluationPlugin`
  emits R-matrix/BWT/FF, `benchmarks` produce task streams. Also `avalanche-rl` fork.
- **Mammoth / PyCIL / LibContinual**: same shape — a trainer + a list of `datasets` +
  plugin callbacks + an R-matrix logger. (github.com/aimagelab/mammoth ; github.com/RL-VIG/LibContinual)

**Recommendation (do NOT rewrite):** keep the wrap-don't-subclass design — it's what makes
"swap LEARN = TRL SFTTrainer" possible. Two concrete upgrades for v1.1:
1. **Make `SFTStrategy` optionally delegate to TRL's `SFTTrainer`** instead of your raw
   PyTorch loop. This is the single highest-credibility change: it converts the pitch from
   "another SFT loop" to "TRL + a continual layer." ~1–2 days.
2. **Expose drift/repair as optional callbacks** (Avalanche-style) so a user can bolt
   VERIFY/REPAIR onto *their own* training loop, not just yours. This is what makes it a
   *framework* rather than a *method*. The interfaces already exist; add a thin callback
   adapter. ~1 day.

---

## E. 2-week scope (Kaggle T4, one person, ~60% already written)

Your `AVR_SECOND_OPINION.md` is the real constraint: AVR v1 works on seed 42 (BWT −0.023,
5.6× less forgetting than naive) but **over-repairs on seed 123** (205 steps ⇒ 99.997%
reversion ⇒ ACC collapse), and subspace repair is dead at r=32 ("the load-bearing subspace
IS the whole 32-dim LoRA update space"). Ship around that reality.

### Ship (the framework + one undeniable demo)
1. **Fix over-repair (1–2 days).** Ship the 10-step interpolation cap you already
   identified (`max_steps_mode: adaptive` + a hard 10 cap). This is a *pragmatic band-aid*,
   not a research breakthrough — be honest about that in the README. Target: seed 123 lands
   ACC≈0.33–0.37, BWT≈−0.02.
2. **Multi-seed TRACE credibility (1 day GPU).** Seeds 42/123/7, LFM2.5-350M. Report mean±std.
   One cross-family run (Qwen2.5-0.5B) to prove it's not LFM-specific.
3. **The gap demo — this is the artifact (2–3 days).** Run the *same* TRACE stream two ways:
   (A) **naive sequential SFT via TRL `SFTTrainer`** (or LEAP) — shows the gap (BWT ≈ −0.13).
   (B) **avr-cl** — shows mitigation (BWT ≈ −0.02). One table, one chart, reproducible on a
   free Kaggle T4. This *is* the proof the gap is real and you fill it.
4. **TRL integration (2 days).** `learn.method: trl_sft` delegates to `SFTTrainer`. Now the
   positioning "TRL but for task streams" is literally true in code.
5. **Package + README + one notebook (2 days).** `pip install avr-cl`, `avr train config.yaml`,
   a Kaggle notebook that reproduces the gap demo in <2h.
6. **Debug buffer (2 days).**

### Cut (explicitly, write as "v2 — research, stubbed")
- DPO/GRPO `LearnStrategy` (stub only — interface reserved, as you have).
- `SubspaceSnapshotInterp` — **keep the stub, don't implement.** Your own second opinion
  proves it fails at r=32/350M. Implementing it in 2 weeks = shipping a known-broken feature.
- **Two-stream hippocampus/neocortex + KL distillation — CUT from the 2-week ship.** It's
  your CL expertise and it's the *research* story, but it is not in your current codebase
  and is high-risk for 2 weeks. Instead: the `Consolidator`/`Oracle` stubs you already have
  *are* the API surface for it. Ship the stubs, put the two-stream in the README's "v2
  research path" as the thing that lands without API changes. That's how you signal research
  depth without betting the ship on it.
- 7B Modal run; Llama/Qwen/Gemma multi-model sweep (keep one cross-family only).

### Smallest demo that proves the gap
4-task TRACE stream, LFM2.5-350M, LoRA r=32. Two columns: **Naive (TRL SFT)** vs **avr-cl**.
Show BWT −0.13 → −0.02 and the per-task R-matrix heatmap (prior tasks collapse under naive,
survive under AVR). Reproducible on Kaggle T4 in <2h. That single figure is the tweet.

---

## F. Naming & positioning

### Naming collisions (verified)
- **`avr-cl` on PyPI → HTTP 404 → AVAILABLE.** Keep it. (`avr` alone is taken; your import
  name `avr` is fine since the distribution is `avr-cl`.)
- `stream-cl`, `streampt`, `continual-pt` also free (404) as backups.
- `stream_framework` (PyPI) is taken but is an unrelated activity-feed/Cassandra lib — no
  LLM collision. `continual-learning` (PyPI) is a personal vision-CL lib — no LLM collision.
- **"ContinualLM" is taken** (UIC-Liu-Lab) — do not rename to anything with "ContinualLM";
  it would collide and confuse. `avr-cl` avoids it.

### Positioning
"TRL but for task streams" is accurate but **commoditizes you** (sounds like a TRL plugin).
Two stronger framings:

1. **"The continual layer for LLM post-training."** TRL/Unsloth/LEAP train one stage; you
   make the *sequence* safe. LEARN→VERIFY→REPAIR as the one-line mental model. This frames
   you as orthogonal-to-and-wrapping the incumbents, not competing with them.
2. **For Liquid specifically:** "LEAP-Finetune gets you one stage. avr-cl gets you a stream
   — without destroying the last one. Validated on LFM2.5-350M."

The differentiator from TRL is *time*: TRL has no concept of "after this task, before the
next." That's the entire gap, stated plainly.

---

## G. Honest verdict (brutal, as requested)

**1. Is the gap real?** **Yes — strongly.** Evidence: (a) zero frameworks in the §A matrix
ship (a)–(e); (b) no `StreamTrainer`/`ContinualTrainer` exists anywhere; (c) ContinualLM is
a 2022-era encoder-LM domain-pretraining methods repo, not a modern decoder post-training
framework; (d) the pain is independently recognized by arxiv (2510.17776), Prime Intellect
("repeatable post-training loop"), Baseten ("sequential SFT on seven tasks → significant
forgetting"), a16z, and the TRACE/CoIN benchmarks. The slot is empty and named.

**2. Minimum 2-week scope.** Fix the 10-step over-repair cap → multi-seed TRACE (42/123/7)
on LFM2.5-350M + one Qwen2.5-0.5B run → **the gap demo (naive TRL SFT vs avr-cl, same
stream)** → `learn.method: trl_sft` integration → PyPI + Kaggle notebook. Cut DPO/GRPO,
subspace repair, two-stream distillation, 7B, multi-model sweep. Ship stubs for the rest.

**3. Smallest proof.** One figure: 4-task TRACE, R-matrix heatmap, Naive (BWT −0.13) vs
avr-cl (BWT −0.02), reproduced on a free T4 in <2h. The heatmap *visually* shows prior-task
collapse under naive and survival under AVR. That is the gap, demonstrated.

**4. The tweet for Liquid AI.**
> TRL/Unsloth/LEAP train your LLM once. Train it again on a new task — it forgets.
> avr-cl is the continual layer for post-training: LEARN → VERIFY (PPL drift) → REPAIR
> (closed-form weight interp). No replay buffer, O(1) memory, no gradients at repair.
> 5.6× less forgetting than naive SFT on TRACE. Built + validated on @LiquidAI LFM2.5-350M.
> Runs on a free Kaggle T4. pip install avr-cl. [gap-demo figure]

Why it works for Liquid: it's on *their* model, names *their* tool (LEAP), and fills a gap
their own README is silent on.

**5. Honest odds of a DM.**
- **Liquid AI:** ~40–55% you get a like/reply from someone on the model/fine-tuning team;
  ~15–25% it converts to a real conversation/DM. Highest-EV because (i) you're on their
  model, (ii) small lab, (iii) the gap is concrete and their own docs expose it.
- **Prime Intellect:** ~20–30% any reaction; they're infra-scale focused and a single-T4
  framework is below their scale lens unless framed as "the CL layer for your repeatable
  post-training loop."
- **Together:** ~10–20%; their open focus is SWE-agent RL, adjacent not core.

**Why it's not higher, honestly:** (a) one-person/2-week artifacts compete with team
frameworks — TRL/verl have dozens of contributors; (b) your own second opinion shows AVR
v1 has a real method limitation (seed-123 over-repair, zero-sum wall at r=32) that a sharp
researcher will probe in any interview — you must lead with the *framework* and be candid
that v1's method is a *pragmatic band-aid, not a SOTA claim*; (c) the CL-methods field is
crowded (RCL/InsCL/SLAO+MVA/CURLoRA/CODYRA/SEE/STABLE) — your moat is the *framework
abstraction + reproducibility + on-their-model*, not "best BWT number."

**The path that actually works:** don't sell "I built the best CL method." Sell "I built
the missing framework — here's the gap proven with a 2-hour reproducible demo on your
model, here's the clean LEARN/VERIFY/REPAIR abstraction that lets *any* method (including
yours, including future two-stream distillation) plug in." That framing is
internship-interview-worthy because it shows research taste (gap identification) + engineering
judgment (scope, honest limitations, pluggable API) + execution (shipped, reproducible).
The DM is a bonus; the artifact is the portfolio piece that earns a referral even without a
cold-DM conversion.

---

## Source index (all URLs referenced)
- TRL: github.com/huggingface/trl , huggingface.co/blog/trl-v1 , huggingface.co/docs/trl
- Unsloth: github.com/unslothai/unsloth , unslothai.substack.com
- Axolotl: github.com/axolotl-ai-cloud/axolotl
- LLaMA-Factory: github.com/hiyouga/LlamaFactory , arxiv.org/abs/2403.13372
- verl: github.com/verl-project/verl , github.com/volcengine/verl/blob/main/docs/start/agentic_rl.rst
- OpenRLHF: github.com/openrlhf/openrlhf , arxiv.org/abs/2405.11143
- SkyRL: github.com/novasky-ai/skyrl , arxiv.org/abs/2511.16108
- prime-rl: github.com/PrimeIntellect-ai/prime-rl , openreview.net/pdf?id=yk3ICpEbv8 , primeintellect.ai/blog/nemotron-3
- verifiers/renderers: github.com/PrimeIntellect-ai/verifiers , github.com/PrimeIntellect-ai/renderers
- AReaL: github.com/areal-project/AReaL , arxiv.org/abs/2505.24298
- OpenPipe ART: github.com/openpipe/art , art.openpipe.ai
- rLLM / DeepSWE: github.com/agentica-project/rllm , together.ai/blog/deepswe
- ContinualLM (the one name-collision): github.com/UIC-Liu-Lab/ContinualLM
- Liquid LEAP-Finetune: github.com/Liquid4All/leap-finetune , docs.liquid.ai
- CL-LLM methods: github.com/mnoorfawi/curlora , github.com/jeff024/codyra , arxiv.org/abs/2510.16089 (STABLE) , aclanthology.org/2025.findings-acl.387 (SEE) , aclanthology.org/2026.mellm-1.30 (Group-Merger)
- Forgetting evidence: arxiv.org/abs/2510.17776 , baseten.co/research/continual-learning , a16z.com/why-we-need-continual-learning
- TRACE benchmark: arxiv.org/abs/2310.06762 , github.com/BeyonderXX/TRACE
- CL-lib patterns: avalanche.continualai.org , github.com/continualai/avalanche-rl , github.com/RL-VIG/LibContinual
- Model merging (overlap): github.com/arcee-ai/mergekit
- Async-RL landscape survey: huggingface.co/blog/async-rl-training-landscape
- HF post-training intern take-home: github.com/huggingface/post-training-takehome
