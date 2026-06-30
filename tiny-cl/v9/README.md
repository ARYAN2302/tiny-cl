# V9: First Honest SLAO vs Naive

The first version to run SLAO and naive in the *same* script, same data, same seed, same model. V9.1 adds module verification (print what LoRA actually wraps) and multi-seed (3 seeds by default).

## SLAO Algorithm (Qiao & Mahdavi, ICLR 2026, arXiv 2512.23017)

```
1. Task 1: standard fine-tune → A_merge=A_1, B_merge=B_1
2. Task i (i>1):
   a. A_init = QR(prev_A)^T (orthogonal rows), B_init = prev_B
   b. Fine-tune both A and B on new task
   c. A_merge = A_ft (replace), B_merge = B_merge + λ(B_ft - B_merge)
      where λ = 1/sqrt(i)
```

## Files

- `v9_slao_controlled.py` — V9.1: the controlled experiment with naive in the same run
- `v9_slao_correct.py` — earlier V9 version (no in-run naive baseline)

## Result

FF(A)=1.09× with rank=32, vs naive FF(A)≈1.50×. This is the first run where the comparison was earned (same data, same seed, same model).

## Honest naming fix

V9.1 prints what LoRA actually wraps: `["in_proj", "out_proj"]` across all 16 layers of LFM2.5-350M (26 LoRA layers total: 20 short-conv + 6 attention `out_proj`). Earlier versions mislabeled this as "conv-only."

## Status

Superseded by v13 (full picture: 5 methods × 5 orderings × 3 seeds).
