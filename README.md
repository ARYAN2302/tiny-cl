# tiny-cl

**AVR (Anchor-Verify-Repair): a closed-form continual-learning method that detects drift via perplexity ratios and repairs it by interpolating LoRA weights toward a snapshot — no replay, no gradients, no labels at repair time.**

Built and evaluated on [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) against the TRACE continual-learning benchmark (4 tasks: C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds). All numbers below are from a single real run, seed 42, TRACE 500-example variant. Raw results JSONs are in [`results/`](results/).

---

## TL;DR

| Method (TRACE, seed 42)        | ACC ↑   | BWT ↑    | FF ↓    | Repair steps |
|--------------------------------|---------|----------|---------|--------------|
| Naive sequential SFT           | 0.379   | −0.130   | 0.130   | —            |
| SLAO + MVA (baseline)          | 0.397   | −0.062   | 0.062   | —            |
| **AVR (this project)**         | 0.374   | **−0.023** | **0.038** | 24 total     |

- **AVR reduces forgetting by 3.4× vs naive and 1.6× vs SLAO+MVA** (on FF / BWT).
- ACC is comparable to naive (0.374 vs 0.379) — AVR trades ~0.5pp of accuracy for 5.6× better retention.
- Repair is **cheap**: 24 closed-form weight interpolations across 4 tasks, no gradient steps.
- The earlier broken AVR variant (v19) hit higher ACC (0.405) but with 2.1× worse forgetting — v23 is the fix.

---

## What AVR is

After training each task `t`:

```
1. Snapshot the LoRA state S_t
2. For each prior task j, compute current PPL on task j's eval set
3. Update ppl_best[j] = min(ppl_best[j], ppl_now[j])
4. VERIFY:  if ppl_now[j] / ppl_best[j] > 1.15  →  task j has drifted
5. REPAIR:  θ ← (1 − α) · θ + α · θ_snapshot       (α = 0.1, closed-form)
            repeat until no task drifts (or max steps)
6. Next task
```

The repair step is a **convex combination in weight space** — no optimizer, no gradients, no labels. It pulls the model toward the snapshot state for drifted tasks. The snapshot is just a LoRA state dict (constant memory, same size as the adapter).

### Repair log from the v23 run

| Task         | Repair steps | Notes |
|--------------|--------------|-------|
| C-STANCE     | 0            | No prior tasks to drift |
| FOMC         | 4            | C-STANCE drifted; repaired in 4 steps |
| NumGLUE-cm   | 10           | Both priors drifted; 10 steps to converge |
| NumGLUE-ds   | 10           | Three priors drifted; 10 steps |
| **Total**    | **24**       | Across the full 4-task stream |

---

## What's mine, what's borrowed

| Component | Source | What I did |
|-----------|--------|------------|
| **AVR** (Anchor-Verify-Repair) | **Mine.** Designed and implemented in this project. | PPL-ratio drift detector + closed-form weight-space repair. This is the contribution. |
| **SLAO + MVA** (baseline) | SLAO: Qiao & Mahdavi, ICLR 2026 (arXiv 2512.23017). MVA: inspired by INTUITOR, Zhao et al., ICLR 2026 (arXiv 2505.19590). | Used as the comparison baseline in `experiments/v18_trace_benchmark.py`. Not my method. |
| **LFM2.5-350M** base model | LiquidAI | Used as-is, frozen, with LoRA adapters (rank 32, `in_proj`+`out_proj`) |
| **TRACE benchmark** | Wang et al. (LLM-CL-Benchmark) | Used as-is for evaluation, 500-example variant |

**My contribution is AVR.** Everything else is either a baseline (SLAO+MVA), a tool (LFM2.5-350M), or an evaluation harness (TRACE).

---

## Results in detail

### v18 — Baselines on TRACE (naive vs SLAO+MVA)

The R matrix `R[i,j]` = accuracy on task `j` after training task `i`. ACC = mean of last row. BWT = backward transfer (negative = forgetting). FF = forgetting factor (positive = forgetting).

**Naive sequential SFT:**

```
R = [[0.465,   —,     —,      —   ],
     [0.380, 0.630,   —,      —   ],
     [0.360, 0.400, 0.358,    —   ],
     [0.295, 0.510, 0.259, 0.450 ]]

ACC = 0.379   BWT = −0.130   FF = 0.130
```

**SLAO + MVA (baseline):**

```
R = [[0.485,   —,     —,      —   ],
     [0.435, 0.580,   —,      —   ],
     [0.395, 0.480, 0.333,    —   ],
     [0.445, 0.545, 0.222, 0.375 ]]

ACC = 0.397   BWT = −0.062   FF = 0.062
```

SLAO+MVA wins on ACC (+1.8pp over naive) and halves forgetting — but still loses 6.2pp of accuracy on earlier tasks by the end of the stream.

### v23 — AVR standalone on TRACE (this project)

```
R = [[0.475,   —,     —,      —   ],
     [0.450, 0.550,   —,      —   ],
     [0.450, 0.570, 0.235,    —   ],
     [0.420, 0.510, 0.259, 0.305 ]]

ACC = 0.374   BWT = −0.023   FF = 0.038
Total repair steps = 24
```

**What this says:**
- BWT = −0.023 is **2.8× better than SLAO+MVA (−0.062)** and **5.6× better than naive (−0.130)**. AVR forgets the least.
- FF = 0.038 is the lowest of all three. Almost no catastrophic forgetting.
- ACC = 0.374 is 2.3pp below SLAO+MVA but **within 0.5pp of naive**. The repair steps preserve old tasks at a small cost to new-task plasticity.
- The R matrix shows AVR holds prior tasks near their peak: C-STANCE stays at 0.42 even after 3 more tasks, vs naive dropping it to 0.295.

### Why AVR's BWT is so much better

AVR's repair is **targeted**: it only fires when a task's PPL drifts >1.15× above its best-seen value, and only on the drifted tasks. SLAO+MVA's merge step is global — it interpolates all B weights every task, which dilutes old tasks structurally (after T tasks, the first task's B carries `∏(1−1/√t) ≈ 12.4%` of its trained magnitude at T=3). AVR doesn't have this dilution because it doesn't merge — it snapshots and conditionally rewinds.

---

## Repository

```
tiny-cl/
├── README.md
├── LICENSE                       MIT
├── requirements.txt
├── CITATION.cff
├── .gitignore
│
├── experiments/
│   ├── v18_trace_benchmark.py   Baseline: naive vs SLAO+MVA on TRACE
│   └── v23_avr_trace.py         AVR standalone on TRACE (this project)
│
└── results/
    ├── v18_trace.json           Raw baseline results (seed 42)
    └── v23_avr.json             Raw AVR results (seed 42)
```

## Run

```bash
pip install -r requirements.txt

# Baselines (naive vs SLAO+MVA) — ~3-4h on T4
python experiments/v18_trace_benchmark.py

# AVR (this project) — ~2h on T4
python experiments/v23_avr_trace.py
```

Each script auto-installs missing deps, downloads TRACE via gdown, and is reproducible with `SEED=42`. Output JSONs match the format in `results/`.

## Config

- **Model:** `LiquidAI/LFM2.5-350M` (frozen, bfloat16)
- **LoRA:** rank 32, alpha 32, dropout 0.05, targets `["in_proj", "out_proj"]` (26 modules across 16 layers)
- **Optimizer:** AdamW, lr 2e-4, wd 0.01, max grad norm 1.0
- **TRACE variant:** `LLM-CL-Benchmark_500` (500 examples per task)
- **AVR hyperparams:** drift threshold 1.15 (PPL ratio), repair α 0.1, max repair steps 100 (cap; never hit)
- **Hardware:** single T4 GPU (Kaggle), 16GB VRAM

## Citation

```bibtex
@misc{tiny-cl-2026,
  author       = {Aryan Thakur},
  title        = {tiny-cl: AVR — Anchor-Verify-Repair for Continual Learning on LFM2.5-350M},
  year         = {2026},
  url          = {https://github.com/ARYAN2302/tiny-cl},
  note         = {AVR is original to this project. Baselines (SLAO, MVA) are from
                  Qiao \& Mahdavi (ICLR 2026) and INTUITOR (ICLR 2026) respectively.
                  Evaluated on the TRACE benchmark (Wang et al.).}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
