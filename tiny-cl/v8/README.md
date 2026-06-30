# V8: O-LoRA Attempt

Tried O-LoRA (orthogonal LoRA, from the ICLR 2025 O-LoRA paper) as an alternative to Anchor-AVR. The initial run had a critical bug: only conv-layer LoRA was being saved/restored, missing attention-layer LoRA. V8.1 fixes the bug and adds a diagnostic.

## Files

- `v8_kaggle.py` — Initial O-LoRA Kaggle cell (has the bug)
- `v8_norm_test.py` — Test orthonormality of A across tasks
- `v8_olora_diagnostic.py` — V8.1: diagnostic that found and fixed the bug

## What we learned

O-LoRA's per-task orthogonal constraint works on paper but underperforms SLAO in practice on LFM2.5-350M. SLAO's "A=replace, B=interpolate" is simpler and gives better forgetting numbers. The diagnostic confirmed that the bug was module-targeting, not algorithmic.

## Status

Superseded by v9 (SLAO).
