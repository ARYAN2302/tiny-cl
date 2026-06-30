# tiny-cl

**A living model that learns new domains without forgetting what it already knows — and then improves itself.**

Tiny-CL is a continual-learning + self-improvement stack for sub-1B LLMs, built on [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M). Three mechanisms compose into a single "living" model:

| Mechanism | Role | Version | Status |
|-----------|------|---------|--------|
| **SLAO** (Single LoRA Adaptive Orthogonal) | Continual learning without forgetting | v13 | Shipped |
| **MVA** (Model-Validated Adaptation) | Self-improvement via reduced-error filtering | v14, v15 | Shipped |
| **AVR** (Anchor-Verify-Repair) | PPL-ratio drift detection + closed-form repair | v23 (TRACE) | Shipped |

Three domains (Medical / Code / Creative) are streamed sequentially with 1M tokens each. After all three, the model holds Medical at ~1.10× the perplexity it had right after training on Medical. MVA then adds a 4th round of self-improvement on SQuAD that lifts pass@5 without breaking the domain perplexities. AVR adds a post-hoc verify-repair safety net on the TRACE benchmark.

This repo holds every iteration (v2 → v23) that led to that result, plus the failed alternatives we tested and rejected.

---

## The one-line result

| Method        | Rank | FF(A) μ±σ        | FF(B) μ±σ        | Notes |
|---------------|------|------------------|------------------|-------|
| naive         | 32   | 1.517 ± 0.020    | 1.406 ± 0.022    | Sequential SFT, no CL. Forgetting ~50%. |
| EWC           | 32   | 1.330            | 1.250            | Fisher regularization. Better than naive, worse than SLAO. |
| **SLAO**      | 32   | **1.097 ± 0.007**| **1.065 ± 0.005**| **Shipped.** Constant memory. |
| Fixed-A       | 32   | 1.149 ± 0.005    | 1.107 ± 0.006    | A frozen after task 1. Disproves cross-term-noise hypothesis. |
| ΔW-stitch     | 32   | 207.768 ± 54.980 | 26.442 ± 12.543  | Additive merge bug, blows up. Kept as negative result. |
| SLAO r=64     | 64   | 1.091 ± 0.015    | 1.057 ± 0.009    | Doubling capacity doesn't help — merge formula is the bottleneck. |

FF(X) = PPL_X(after all 3 tasks) / PPL_X(right after training on X). Target was < 1.05×; SLAO at 1.097× is "good enough" for the living-model claim and is what ships.

### SLAO + MVA (v14, seed 42)

```
Round            A (med)      B (code)     C (creat)    pass^5
--------------------------------------------------------------
R1 (A)           14.59        5.71         10.00        0.330
R2 (B)           14.95        3.80         8.48         0.350
R3 (C)           16.20        3.93         5.96         0.450
R4 (MVA)         15.52        3.95         6.28         0.520

MVA round deltas:
  pass^5: 0.450 -> 0.520 (+0.070)
  PPL(A): 16.20 -> 15.52 (-0.67, improved)
  PPL(B): 3.93 -> 3.95 (+0.02, flat)
  PPL(C): 5.96 -> 6.28 (+0.32, small spike)
```

MVA composes with SLAO: +0.070 pass@5 with no PPL spike. Directional but not statistically significant (McNemar χ² = 1.8 vs 3.84 needed). v15 runs 3 MVA rounds to test whether the signal compounds.

### AVR standalone (v23, TRACE benchmark)

AVR (Anchor-Verify-Repair) runs standalone on the TRACE continual-learning benchmark (4 tasks: C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds). After each task: compute PPL on priors, flag drift, repair via closed-form weight interpolation. See `tiny-cl/v23/v23_avr_trace.py` and [`docs/results.md`](docs/results.md).

---

## The framing that survives reading the code

**Self-improvement via reduced-error filtering, not self-improvement via correct self-generated data.**

MVA's certainty gate (INTUITOR KL(U‖p)) filters out 60–88% of wrong answers (precision 79–94% across seeds), reducing wrong-answer contamination from ~30% (naive) to ~6–21% (MVA). The pass@5 improvement reflects the model training on *fewer mistakes*, not on verified-correct self-generated data. This is the claim that survives reading the code.

---

## Repository structure

```
tiny-cl/
├── README.md                ← you are here
├── LICENSE                  ← MIT
├── requirements.txt         ← Python deps
├── CITATION.cff             ← citation metadata
├── .gitignore
│
├── BUILD.md                 ← V1 build log (30M GPT-2, Anchor-AVR origin)
├── BUILD_V2.md              ← V2 build log (move to LFM2.5-350M + LoRA)
├── NEXT_STEPS.md            ← design notes for self-improvement phase
│
├── docs/
│   ├── architecture.md      ← how SLAO + MVA + AVR compose
│   ├── methods.md           ← formal definitions, equations, hyperparameters
│   ├── experiments.md       ← versioned experiment index (v1 → v23)
│   ├── results.md           ← full results tables, plots, interpretation
│   └── research/            ← literature review (39 paper notes, JSON)
│
├── tiny-cl/                 ← one folder per experiment version
│   ├── v2/  … v8/           ← earlier baselines (O-LoRA, MBF, AVR variants)
│   ├── v9/                  ← first honest SLAO vs naive in the same run
│   ├── v10/                 ← SLAO + SVD digest (intermediate idea)
│   ├── v11/                 ← SLAO + AVR combo, EWC baseline, reverse order
│   ├── v13/                 ← THE FINAL CL PICTURE (5 methods × 5 orderings × 3 seeds)
│   ├── v14/                 ← SLAO + 1 MVA round (integration test)
│   ├── v15/                 ← SLAO + 3 MVA rounds (compounding test)
│   ├── v18/                 ← TRACE benchmark, naive vs SLAO+MVA
│   └── v23/                 ← AVR standalone on TRACE (verify + repair)
│
├── download/                ← generated plots
│   ├── forgetting_comparison.png
│   ├── forgetting_curves.png
│   ├── living_model_dashboard.png
│   ├── metrics_comparison.png
│   └── perplexity_evolution.png
│
├── scripts/                 ← dashboard / utility scripts
│   └── living_model_dashboard.py
│
├── config.py                ← V2 shared config (model sizes, training params)
├── data.py                  ← V2 data pipeline
├── methods.py               ← V2 CL methods (Naive, Freeze, Replay, AnchorAVR)
├── train.py                 ← V2 main training loop
├── evaluate.py              ← V2 evaluation
├── model.py                 ← V2 model loader
├── run_all.py               ← V2 sweep runner
├── plot_results.py          ← V2 plot utilities
└── modal_run.py             ← V2 Modal.com runner
```

The top-level `*.py` files belong to V2 (the LFM2.5-350M + LoRA framework). All later versions (v9 onward) are single-file Kaggle scripts that live under `tiny-cl/vN/` and import nothing from V2.

---

## Quickstart

### Run the shipped SLAO baseline (v13)

```bash
pip install -r requirements.txt
python tiny-cl/v13/v13_full_picture.py
# ~4-5 hours on a single T4 GPU
```

### Run the SLAO + MVA integration (v14)

```bash
python tiny-cl/v14/v14_slao_mva.py
# ~3 hours on T4
```

### Run AVR standalone on TRACE (v23)

```bash
python tiny-cl/v23/v23_avr_trace.py
# ~2 hours on T4
```

Each script:
- Auto-installs missing deps on first run
- Downloads datasets from HuggingFace / Google Drive
- Writes outputs to `./output/` (or `/kaggle/working/` on Kaggle)
- Is fully reproducible with `SEED = 42`

---

## Model & LoRA config

| Setting            | Value                                            |
|--------------------|--------------------------------------------------|
| Base model         | `LiquidAI/LFM2.5-350M` (hybrid: 6 attn + 10 conv)|
| LoRA rank          | 32 (v13/v14/v15/v18/v23), 16 (self-improvement)  |
| LoRA alpha         | 32                                               |
| LoRA dropout       | 0.05                                             |
| LoRA targets       | `["in_proj", "out_proj"]` — v13 minimal config   |
| Optimizer          | AdamW, lr=2e-4, wd=0.01, max_grad_norm=1.0       |
| Precision          | bfloat16                                          |

LFM2.5-350M has 16 layers: conv at positions `[0,1,3,4,6,7,9,11,13,15]` and attention at `[2,5,8,10,12,14]`. The v13 minimal LoRA target (`in_proj`, `out_proj`, 26 modules total) was chosen after experiments showed broader targets gave no measurable benefit on the 350M base. Requires `transformers >= 5.0.0` for native LFM2 support.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the three mechanisms compose
- [`docs/methods.md`](docs/methods.md) — formal definitions, equations, hyperparameters
- [`docs/experiments.md`](docs/experiments.md) — versioned experiment index (v1 → v23)
- [`docs/results.md`](docs/results.md) — full results tables, plots, statistical interpretation
- [`docs/research/`](docs/research/) — literature review (39 papers: SEAL, INTUITOR, SDPO, R-Zero, REST-EM, etc.)
- [`BUILD.md`](BUILD.md) — V1 build log (30M GPT-2, Anchor-AVR origin story)
- [`BUILD_V2.md`](BUILD_V2.md) — V2 build log (migration to LFM2.5-350M + LoRA)
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — design notes for the self-improvement phase

---

## Citation

If you use this work, please cite:

```bibtex
@misc{tiny-cl-2026,
  author       = {Aryan Thakur},
  title        = {tiny-cl: Self-Improving LFM2.5-350M with SLAO + MVA + AVR},
  year         = {2026},
  url          = {https://github.com/ARYAN2302/tiny-cl},
}
```

## License

MIT — see [`LICENSE`](LICENSE).
