# V10: SLAO + SVD Digest

After SLAO merge, the LoRA product has redundant directions. SVD digest recompresses into optimal rank-r factors, keeping only the highest-energy directions. The model "digests" after each meal — absorbs what matters, drops what doesn't.

## Algorithm

```
1. Train → SLAO merge (A=replace, B=interpolate)
2. SVD: B@A = U@S@V^T → keep top-r → new B, new A
3. A now has orthogonal rows (free ortho init for next task!)
```

## Files

- `v10_living_model.py` — single-file Kaggle cell

## Result

SVD digest did not measurably beat plain SLAO. The "free ortho init for next task" insight was folded into v13's SLAO implementation (which already does QR re-orthonormalization on A).

## Status

Intermediate idea. Superseded by v13.
