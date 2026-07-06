# tiny-cl

AVR is a continual learning method for small language models. It detects when a model has forgotten an old task by checking perplexity, then repairs the damage by interpolating the weights back toward a saved snapshot. No replay buffer, no gradients at repair time, no reinforcement learning.

I built this because most "continual learning" in LLMs is just absorption. You fine-tune on task B, the weights shift, task A degrades. That's not learning — that's overwriting. Real learning means the model can verify it hasn't broken anything and fix it if it has. AVR does the verify and fix steps in closed-form math.

## The method

After training on each task in a sequence:

1. **Snapshot** the LoRA weights.
2. **Verify**: compute perplexity on each prior task's eval set. If `PPL_now / PPL_best > 1.15`, that task has drifted.
3. **Repair**: `θ ← (1−α)·θ + α·θ_snapshot` where α=0.1. Repeat until drift resolves or you hit the step cap.

The repair step is a convex combination in weight space. No optimizer, no gradients, no labels. It pulls the model toward the snapshot state for drifted tasks. The snapshot is a LoRA state dict — same size as the adapter, constant memory.

## Results

TRACE benchmark, 4 tasks (C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds), LFM2.5-350M + LoRA r=32, seed 42, 500-example variant:

| Method | ACC | BWT | FF | Repair steps |
|---|---|---|---|---|
| Naive sequential SFT | 0.379 | −0.130 | 0.130 | — |
| SLAO + MVA (ICLR 2026) | 0.397 | −0.062 | 0.062 | — |
| **AVR (this repo)** | **0.374** | **−0.023** | **0.038** | 24 |

AVR forgets less than both baselines. ACC is within 0.5pp of naive — a small accuracy cost for 5.6× better retention. Raw results are in `results/`.

The two-stream extension (separate repo: [Living-Model](https://github.com/ARYAN2302/Living-Model)) pushes BWT positive: +0.017 to +0.107 across seeds. That's old tasks *improving* as new ones are learned.

## What's mine, what's borrowed

| Component | Source |
|---|---|
| AVR (Anchor-Verify-Repair) | Mine. Designed and implemented here. |
| SLAO + MVA baseline | Qiao & Mahdavi, ICLR 2026. INTUITOR, Zhao et al., ICLR 2026. |
| LFM2.5-350M | LiquidAI, frozen base. |
| TRACE benchmark | Wang et al. (LLM-CL-Benchmark). |

## Repository

```
tiny-cl/
├── avr/
│   ├── framework.py         # LEARN → VERIFY → REPAIR orchestrator
│   ├── strategy.py          # AVRStrategy — builds trainer from config
│   ├── operators.py         # SnapshotInterp (v1), SubspaceSnapshotInterp (v2 stub)
│   ├── detectors.py         # PPLRatioDetector (v1), KL/Hessian (v2 stubs)
│   ├── trainer.py           # SFTStrategy, ReplaySFTStrategy
│   ├── metrics.py           # R-matrix, BWT, FF, ACC
│   ├── data.py              # TRACE, MMLU stream loaders
│   ├── cli.py               # `avr train config.yaml`
│   └── configs/             # YAML configs
├── experiments/
│   ├── v18_trace_benchmark.py   # Baselines: naive vs SLAO+MVA
│   └── v23_avr_trace.py         # AVR standalone
└── results/
    ├── v18_trace.json
    └── v23_avr.json
```

## Run

```bash
pip install -r requirements.txt

# Baselines (naive vs SLAO+MVA) — ~3-4h on T4
python experiments/v18_trace_benchmark.py

# AVR — ~2h on T4
python experiments/v23_avr_trace.py
```

Each script auto-installs missing deps, downloads TRACE via gdown, and is reproducible with `SEED=42`.

## Config

```yaml
model:
  id: LiquidAI/LFM2.5-350M
  lora_targets: [in_proj, out_proj]
  lora_rank: 32

stream:
  benchmark: trace
  tasks: [C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds]
  seed: 42

learn:
  method: sft
  epochs: 3
  lr: 2e-4

verify:
  detector: ppl_ratio
  threshold: 1.15
  probe_samples: 200

repair:
  operator: snapshot_interp
  alpha: 0.1
  alpha_decay: sqrt
  max_steps_mode: adaptive
```

## What doesn't work yet

- Full 8-task TRACE. Tried on 7B, OOM'd. Need to fix memory.
- Subspace repair (v32). At LoRA r=32, the load-bearing subspace doesn't separate from the orthogonal one. The math doesn't have room to work.
- PPL-gated consolidation (v33). PPL drift fires constantly during training — wrong signal for gating. Documented in the Living-Model repo.

## License

MIT. AVR is original to this project. Baselines and benchmarks are cited above.
