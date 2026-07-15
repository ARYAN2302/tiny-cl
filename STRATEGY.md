# avr-cl: Framework Positioning & Launch Strategy

**Goal**: Position avr-cl as the post-training framework that solves catastrophic forgetting — not just a research artifact, but something practitioners actually use.

**Constraint**: No infinite compute. After TRACE (exp3), we stop running experiments and shift to packaging + distribution.

---

## Part 1: What the Field Looks Like (and Where avr-cl Fits)

### The competition (as of mid-2025)

| Framework / Method | What it does | Stars / Adoption | avr-cl's wedge |
|---|---|---|---|
| **HuggingFace PEFT** | LoRA, adapters, IA3 | ~16k stars, default standard | PEFT has no CL concept. avr-cl wraps PEFT and adds VERIFY+REPAIR. |
| **TRL** | SFT/DPO/PPO training loops | ~10k stars, default for RLHF | TRL has no "after this task, did I forget?" check. avr-cl plugs into TRL's training loop. |
| **Axolotl** | Config-driven fine-tuning | ~8k stars, popular with practitioners | Axolotl does single-task fine-tuning. avr-cl does sequential multi-task with forgetting protection. |
| **Unsloth** | Fast/cheap fine-tuning | ~25k stars, viral for speed | Unsloth optimizes single training runs. avr-cl optimizes *sequential* runs — the gap Unsloth doesn't address. |
| **mergekit** | Post-hoc model merging | ~5k stars, Arcee AI backs it | mergekit fixes damage *after* the fact. avr-cl prevents it *during*. Different problem, complementary tools. |
| **O-LoRA / SLAO / GORP** | Academic CL methods | papers + thin repos, low adoption | These are methods, not frameworks. No `pip install`, no docs, no community. avr-cl is the framework that makes CL accessible. |

### The gap avr-cl fills

**Nobody ships a usable continual learning framework for LLMs.** The academic methods (O-LoRA, SLAO, GORP, EWC) exist as paper code — jupyter notebooks, hardcoded paths, no docs. Practitioners who want to fine-tune sequentially either:
1. Naively SFT and accept forgetting (most common)
2. Retrain from scratch (expensive)
3. Use replay buffers (privacy/plumbing nightmare)
4. Try to implement EWC from a paper (rare, error-prone)

avr-cl is the first thing you can `pip install` and actually use. That's the positioning.

### The unique angle (repeat this everywhere)

> **"Every CL method tries to *prevent* forgetting. avr-cl *detects* it and *repairs* it."**

This is the one-sentence pitch. The LEARN → VERIFY → REPAIR loop is the mental model. The VERIFY step is the innovation — nobody else checks if the model forgot. Every blog post, tweet, and README should hammer this.

---

## Part 2: The Packaging Audit (What's Broken Now)

### Current state of the repo

**What works:**
- Core `avr/` package (634 LOC, 8 files, clean modular API)
- `avr.run()` — one-call API for the full LEARN→VERIFY→REPAIR loop
- Swappable phases (custom repair_fn, custom scorer)
- Headline result (Qwen3-1.7B, BWT -0.078 vs -0.453)
- EWC baseline implemented
- Cross-domain result (exp4)

**What's broken or missing:**

| Issue | Impact | Fix effort |
|---|---|---|
| Not on PyPI | `pip install avr-cl` doesn't work | 1 day (configure GitHub Actions) |
| No documentation site | README only, no API docs | 2-3 days (mkdocs + API reference) |
| No quickstart notebook | New users can't try it in 5 min | 1 day (Google Colab notebook) |
| README claims are scattered | Results not consolidated | 1 day (results table in README) |
| No HuggingFace Hub integration | Can't push repaired models to Hub | 1-2 days (add `push_to_hub`) |
| No CLI examples | `avr train config.yaml` undocumented | 1 day (CLI docs + examples) |
| Test coverage = 0 | Refactoring is risky | 2-3 days (pytest, smoke tests) |
| No CI/CD | Regressions slip in | 1 day (GitHub Actions: lint + test) |
| Dependency on modelscope for downloads | Fragile on Kaggle | Already fixed (direct HTTP) |

### The developer experience gap

A new user's current journey:
1. Find the repo (good README, good hero image)
2. Read the pitch (compelling)
3. Try `pip install git+https://...` (works, but feels unofficial)
4. Try to use it... no examples beyond the README snippet
5. Look at scripts/ — 7 experiment scripts, all 500+ lines, all Kaggle-specific
6. Give up or file an issue

**The journey should be:**
1. Find the repo
2. `pip install avr-cl`
3. Open the quickstart Colab
4. Run 5 lines of code on their own model + data
5. See the repair loop fire
6. Share it

---

## Part 3: The 4-Week Launch Plan

### Week 1: Make it pip-installable and usable (the foundation)

**Goal**: A practitioner can `pip install avr-cl` and run a working example in under 10 minutes.

**Tasks:**

1. **Publish to PyPI** (Day 1-2)
   - Set up `pyproject.toml` properly (already mostly done)
   - Create GitHub Actions workflow to auto-publish on tag push
   - First release: `v0.1.0` — tag and push
   - Verify `pip install avr-cl` works in a clean environment

2. **Write a 5-minute quickstart** (Day 2-3)
   - Google Colab notebook: load a small model (Qwen3-0.6B), run 2-task stream, show repair firing
   - Should run on free Colab T4, no Kaggle needed
   - Embed in README as a "Try it now" button
   - This is the single most important asset — make it bulletproof

3. **Write proper docs** (Day 3-5)
   - Use mkdocs-material (looks professional, easy to maintain)
   - Sections: Quickstart, API Reference, How It Works, Examples, Benchmarks
   - Host on GitHub Pages (free)
   - API reference auto-generated from docstrings (use mkdocstrings)

4. **Add HuggingFace Hub integration** (Day 5-7)
   - `avr.push_to_hub(model, repo_id)` — push the repaired model to HF Hub
   - This makes avr-cl part of the HF ecosystem, not a side tool
   - Critical for adoption — practitioners live on HF Hub

**Deliverable**: `pip install avr-cl` works, docs site is live, quickstart Colab runs.

### Week 2: Tell the story (the launch)

**Goal**: People know avr-cl exists and understand why it matters.

**Tasks:**

1. **Write the launch blog post** (Day 8-10)
   - Title: "Your fine-tune silently broke your model. Here's how to fix it."
   - Structure: The problem (with the heatmap image) → The insight (VERIFY is missing) → The method (LEARN→VERIFY→REPAIR) → The results (table) → Try it (Colab link)
   - Publish on: your blog, Medium, HuggingFace blog (submit to HF), r/LocalLLaMA, HackerNews
   - This is the single piece of content that drives everything else

2. **Make the hero artifact** (Day 10-11)
   - The heatmap image is good, but make it better: an animated GIF showing the repair loop firing in real-time
   - Before/after: Naive (tasks collapse) vs AVR (tasks survive)
   - This is what gets shared on Twitter/LinkedIn

3. **Social media push** (Day 11-12)
   - Twitter/X thread: 10 tweets walking through the problem → solution → results → Colab link
   - LinkedIn post: same content, different framing (more business-oriented)
   - Tag: @HuggingFace, @weights_biases, key ML influencers
   - Post in r/MachineLearning, r/LocalLLaMA, r/MLQuestions

4. **HuggingFace Spaces demo** (Day 12-14)
   - A Gradio app: user picks 2 tasks, watches the repair loop fire in real-time
   - Runs on a free HF Space (small model, small data)
   - This is the "try before you install" gateway

**Deliverable**: Blog post live, social media pushed, HF Spaces demo running.

### Week 3: Build credibility (the evidence)

**Goal**: avr-cl is taken seriously by researchers and practitioners.

**Tasks:**

1. **Consolidate all results into a benchmarks page** (Day 15-16)
   - One page in the docs: every experiment, every model, every metric
   - Format: model × task stream × method → BWT, ACC, repairs
   - Include the TRACE result (from exp3)
   - Compare against published baselines (GORP, O-LoRA, EWC)
   - This is what researchers link to

2. **Write the arXiv preprint** (Day 16-19)
   - Title: "avr-cl: Detect and Repair Catastrophic Forgetting in Sequential LLM Fine-Tuning"
   - Structure: Introduction → Related Work → Method → Experiments → Analysis → Conclusion
   - 8-10 pages, NeurIPS/ICLR format
   - Submit to arXiv — gives you a citation target
   - Even if you don't submit to a venue, the arXiv paper is the "this is serious" signal

3. **Record a 5-minute demo video** (Day 19-21)
   - Screen recording: pip install → quickstart → watch repair fire → results
   - Upload to YouTube, embed in README and blog post
   - This is what gets shared in Slack/Discord communities

**Deliverable**: arXiv preprint, benchmarks page, demo video.

### Week 4: Build the ecosystem (the moat)

**Goal**: avr-cl is integrated into other tools, making it hard to displace.

**Tasks:**

1. **Integrate with TRL** (Day 22-24)
   - Open a PR or issue on HuggingFace TRL: "Add continual learning support via avr-cl"
   - Even a partial integration (e.g., a `CLTrainer` class that wraps `SFTTrainer`) gets you into the HF ecosystem
   - This is the highest-leverage integration — TRL users are exactly your audience

2. **Add integration examples** (Day 24-25)
   - Example: avr-cl + Unsloth (fast training + CL protection)
   - Example: avr-cl + Axolotl (config-driven CL)
   - Example: avr-cl + mergekit (CL during training, merging after)
   - Each example is a blog post + notebook

3. **Set up a Discord/GitHub Discussions** (Day 25-26)
   - GitHub Discussions for Q&A (lower friction than Discord)
   - Pin: quickstart, benchmarks, roadmap
   - Respond to every issue within 24 hours for the first month

4. **Write the roadmap** (Day 26-28)
   - Public roadmap in the repo: what's next (DPO support, 7B+ models, more benchmarks)
   - This signals the project is active and has a future
   - Invites contributors

**Deliverable**: TRL integration started, integration examples, active community channels.

---

## Part 4: The Positioning Statement (use this verbatim)

**For** ML practitioners and researchers fine-tuning LLMs sequentially
**who** need to add new capabilities without breaking existing ones,
**avr-cl** is a continual post-training framework
**that** detects and repairs catastrophic forgetting after each training stage.
**Unlike** EWC, replay buffers, or O-LoRA (which try to prevent forgetting blindly),
**avr-cl** verifies whether forgetting occurred and repairs it in weight space — no replay buffer, no old training data, one LoRA snapshot.

**One-line pitch**: *"Your fine-tune silently broke your model. avr-cl tells you, and fixes it."*

**Three-word summary**: Detect. Repair. Continue.

---

## Part 5: What NOT to Do

1. **Don't run more experiments before launching.** The TRACE result + Qwen3 headline + cross-domain result is enough evidence. More experiments = more delay. Ship.

2. **Don't build a web app.** A Gradio demo on HF Spaces is fine. A full web app is a distraction. The product is the Python library.

3. **Don't chase GitHub stars.** Stars don't mean adoption. Downloads from PyPI, Colab opens, and arXiv citations do. Optimize for those.

4. **Don't compare to mergekit.** mergekit is a different tool (post-hoc merging). Position as complementary, not competitive. "Use mergekit after, avr-cl during."

5. **Don't claim it works on all models.** Be honest: validated on 1.7B (Qwen3, LFM2.5). 7B+ is roadmap. Honesty builds trust; overclaiming destroys it.

6. **Don't skip the arXiv paper.** Even if you never submit to a venue, the arXiv preprint is the credibility signal that separates toys from tools.

7. **Don't wait for perfection.** v0.1.0 with rough edges > v1.0.0 that never ships. The quickstart Colab working is more important than the API being perfect.

---

## Part 6: Success Metrics (30 days post-launch)

| Metric | Target | How to measure |
|---|---|---|
| PyPI downloads | 1,000+ | `pypistats.org/packages/avr-cl` |
| GitHub stars | 500+ | GitHub insights |
| Quickstart Colab opens | 2,000+ | Colab view count |
| HF Spaces demo visits | 1,000+ | HF analytics |
| arXiv citations | 5+ | Google Scholar |
| Blog post views | 10,000+ | Analytics |
| Issues/PRs from external users | 10+ | GitHub |
| TRL integration discussion | Opened | HF repo |

If you hit 30% of these, you have a real framework. If you hit 50%, you have momentum. If you hit 80%, you have a category leader.

---

## Part 7: The Honest Assessment

**What avr-cl has going for it:**
- A genuinely novel angle (VERIFY is the missing step)
- A clean, modular codebase (634 LOC, swappable phases)
- A compelling headline result (5.8× less forgetting)
- A real problem (everyone fine-tuning sequentially hits this)

**What it's up against:**
- HuggingFace ecosystem inertia (PEFT + TRL are defaults)
- "Just retrain from scratch" is the path of least resistance
- Academic CL methods have low awareness among practitioners
- The framework space is crowded (Axolotl, Unsloth, TRL, PEFT...)

**The bet:**
Practitioners will adopt avr-cl if (and only if) the 5-minute quickstart works and the story is clear. The technology is sound. The packaging is the bottleneck. Fix the packaging, and the framework spreads.

---

## Immediate Next Action (do this today)

1. **Create a GitHub issue titled "v0.1.0 release checklist"** with the Week 1 tasks
2. **Set up the PyPI publish workflow** (GitHub Actions — I can generate this)
3. **Draft the quickstart Colab** (I can generate a skeleton)
4. **Pick a launch date** — 4 weeks from now. Work backwards.

The experiments are done. The framework works. Now it's about making it usable and telling people. Ship it.
