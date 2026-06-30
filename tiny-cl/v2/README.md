# V2: The Living Model (LFM2.5-350M + LoRA)

The migration from V1 (30M GPT-2 from scratch) to a pretrained LFM2.5-350M base with LoRA adapters. This is the framework that all later single-file Kaggle scripts (v4 onward) build on.

## What changed from V1

| | V1 | V2 |
|---|---|---|
| Model | 30M GPT-2 (from scratch) | LFM2.5-350M (pretrained, frozen base) |
| Training | Train from random weights | Fine-tune via LoRA |
| Domains | Stories → Wiki → News | Medical → Code → Creative Writing |
| Key mechanism | Anchor-pull on all weights | Anchor-pull on LoRA adapter weights only |
| Framing | "Continual learning" | "Continuous Absorption" — the model is alive |

## Files

| File             | Role                                                    |
|------------------|---------------------------------------------------------|
| `config.py`      | Model sizes, training params, method configs            |
| `data.py`        | Download + tokenize + split data into 3 phases          |
| `model.py`       | LFM2.5-350M loader (with bf16 + device_map)             |
| `methods.py`     | CL methods: Naive, Freeze, Replay, AnchorAVR            |
| `train.py`       | Main training loop (single experiment)                  |
| `evaluate.py`    | Per-domain perplexity evaluation                        |
| `run_all.py`     | Sweep runner (all methods × seeds)                      |
| `plot_results.py`| Plot utilities (forgetting curves, PPL evolution)       |
| `modal_run.py`   | Modal.com remote runner (optional)                      |
| `smoke_test.py`  | 1-minute sanity check                                   |
| `anchors.py`     | Anchor snapshot management (save/load/compute)          |
| `requirements.txt` | V2-only deps (the root `requirements.txt` is newer)   |

## Why V2 exists

V1 proved Anchor-AVR works on a 30M model trained from scratch (3× less forgetting than all baselines). V2 asks the harder, more important question:

**Can we take an existing pretrained model and make it continual?**

This is what the industry actually needs. No one trains from scratch. Everyone fine-tunes pretrained models. And every fine-tune destroys old knowledge. V2 fixes that.

## Status

Superseded by the single-file Kaggle scripts (v4 onward) for portability. The V2 framework is kept because it's the cleanest reference implementation: each concern (data, methods, training, eval) is in its own file, and you can swap methods by editing `config.py` without touching `train.py`.

## See also

- [`docs/BUILD.md`](../../docs/BUILD.md) — V1 build log (30M GPT-2, Anchor-AVR origin)
- [`docs/BUILD_V2.md`](../../docs/BUILD_V2.md) — full V2 build log
- [`../../docs/architecture.md`](../../docs/architecture.md) — how V2's framework evolved into the three-mechanism living model
