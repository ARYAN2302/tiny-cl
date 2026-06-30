# V6: Diagnostics

Diagnostic notebooks and scripts that investigate why V5's certainty gate lets confident-wrong answers through (the 6–21% wrong-answer contamination that defines the "reduced-error filtering" framing).

## Files

- `v6_diagnostic.py` — Per-question certainty-vs-correctness scatter, wrong-answer audit
- `v6_kaggle.py` — Kaggle cell version
- `v6_kaggle_notebook.ipynb` — Jupyter notebook version with inline plots

## What we learned

- The certainty signal has high variance on wrong answers: some wrong answers get high certainty (confident-wrong), and the gate cannot distinguish them from confident-correct.
- The 79–94% precision ceiling is structural — a single-signal gate cannot do better without an additional correctness signal (which would defeat the "no external reward" purpose).
- This led to the **honesty clause** in v14/v15: MVA is *reduced-error filtering*, not *correct-data generation*.

## Status

Diagnostic only. Results folded into v14's framing.
