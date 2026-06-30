# V4: First Kaggle Port

Moves V2's framework to a single Kaggle cell so it runs on free T4 GPUs. Includes V2-style Python files plus a `v4_kaggle.py` single-cell script.

## Files

- `v4_kaggle.py` — single-file Kaggle cell (the port)
- `BUILD.md` — V4 build notes
- V2 framework: `anchors.py`, `config.py`, `data.py`, `evaluate.py`, `methods.py`, `models.py`, `plot_results.py`, `train.py`

## Status

Superseded by V5+ (cleaner single-file scripts that drop the V2 framework dependency).
