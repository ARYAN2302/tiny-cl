# Next Steps — Self-Improvement Without Forgetting

The continual-learning core is shipped: SLAO r=32 on LFM2.5-350M holds three domains at FF(A) = 1.097×. The CL core is now frozen. We do not touch it. Everything below builds on top of it.

---

## The goal

A living model that, in production, gets better from its own usage:

- The model generates responses on-device.
- The user gives binary feedback (accept / reject, like Cursor Tab's accepted-vs-discarded suggestions).
- Accepted responses become training signal.
- The model updates itself via SLAO so it never forgets prior domains.

No server round-trip for the update. No growing memory. No task labels at inference. This is the unmet market gap.

---

## The combination (and why it's novel)

| Component | Source | Role |
|-----------|--------|------|
| SLAO merge | Qiao & Mahdavi, ICLR 2026 | Constant-memory continual learning — already shipped. |
| SDFT (Self-Distillation FT) | Lin et al., 2024 | On-policy distillation: model generates, model teaches itself from its own good outputs. No labels needed. |
| Binary user feedback | Cursor Tab pattern | Filter which self-generations become training signal. |
| LoRA r=32 | standard | Same as the CL core — keeps update cheap and NPU-friendly. |
| LFM2.5-350M | LiquidAI | Sub-1B hybrid conv+attention. Already our base. |

Nobody ships this combination. The closest neighbours:

- **Llama-Factory / Unsloth / Axolotl** — fine-tuning frameworks, no continual learning, no self-improvement loop.
- **LoRAHub** — composes LoRAs but memory grows with task count.
- **SEAL** (self-improvement via RL) — research, 30-45s latency per update, not mobile-feasible.
- **QVAC Fabric** (Tether) — on-device LoRA FT, but no CL, no self-improvement.
- **Apple PLUM** — synthetic QA from accepted interactions, then standard SFT. No CL either.

The novel claim: **SDFT + binary feedback + SLAO + sub-1B LoRA = the first on-device self-improving living model.**

---

## The risks (in order)

### Risk 1: SDFT fails below 3B without a bootstrap
Lin et al. report SDFT works above 3B parameters. Below 3B, the model's own generations aren't good enough to teach itself — the teacher and student are both weak. LFM2.5-350M is sub-1B.

**Mitigation:** Bootstrap with a small high-quality SFT pass (10-50k tokens of golden data) before turning on SDFT. The SFT pass lifts generation quality above the SDFT viability threshold; from there the loop is self-sustaining.

**Fallback:** If SDFT still fails after bootstrap, switch to PLUM-style synthetic QA pairs (use a stronger teacher — GPT-4o-mini or similar — to expand accepted interactions into QA pairs), then standard LoRA SFT through SLAO. More expensive, but the CL core stays the same.

### Risk 2: Binary feedback is too sparse
Most users won't bother to accept/reject. If <5% of interactions get feedback, the loop starves.

**Mitigation:** Implicit feedback first, explicit second. Accept = the generation was kept (copied, used in next prompt, dwell time above threshold). Reject = discarded within 2s. This pushes feedback rate to ~30-50% of interactions.

### Risk 3: SDFT amplifies the model's biases
If the model is wrong about something and generates confidently, SDFT teaches it to be more confidently wrong. Self-distillation has no external ground truth.

**Mitigation:** Two-layer filter: (i) only accept generations where the model's own confidence is in a mid-band (not too high — likely memorized; not too low — likely wrong), (ii) cross-check against the user's binary feedback before adding to the SDFT buffer.

### Risk 4: Update latency on device
SLAO merge is cheap (one QR + one interpolation per layer). SDFT requires forward passes on accepted generations. On a phone NPU, this could be 5-15s per accepted sample.

**Mitigation:** Batch updates overnight on charger. Don't block the user. Queue accepted generations and run the SLAO+SDFT merge when the device is idle and plugged in.

### Risk 5: Forgetting across the self-improvement boundary
Even with SLAO, if the model trains on user-specific feedback for weeks, it may drift from the general distribution and lose calibration on out-of-distribution inputs.

**Mitigation:** Periodic KL anchor to the shipped base model — every N updates, add a small KL(base || current) term to the loss. SLAO handles the weight merging; the KL anchor handles the distribution drift. This is essentially AVR (which we already tested in V11) applied to the self-improvement loop.

---

## The 6-week plan

### Week 1: SDFT viability on sub-1B
Reproduce Lin et al.'s SDFT on LFM2.5-350M with a single domain (Medical). Compare:
- (a) Plain SFT on the dataset.
- (b) SDFT with no bootstrap.
- (c) SDFT with a 10k-token bootstrap.

If (c) reaches within 5% of (a)'s PPL, SDFT is viable. If not, switch to PLUM fallback.

### Week 2: SDFT + SLAO stack
Run SDFT through the SLAO merge on 3 sequential domains. The CL core stays untouched; SDFT replaces the SFT step inside each task. Confirm FF(A) stays around 1.10×.

### Week 3: Binary feedback filter
Build a synthetic feedback simulator (we don't have real users yet). Generate responses, score them with a held-out judge model (GPT-4o-mini), binarize to accept/reject. Feed accepted responses into SDFT.

### Week 4: pass^k measurement
pass^k = probability the model succeeds across ALL k trials. Unlike pass@k (at least one success), pass^k penalizes inconsistency. A living model needs high pass^k, not just high pass@k — drift across updates would tank pass^k even if average quality goes up. Build the eval harness.

### Week 5: On-device packaging
Wrap the SLAO+SDFT+feedback loop in the QVAC Fabric on-device path (or a minimal equivalent). Target: <30s per accepted-sample update on a phone NPU.

### Week 6: End-to-end demo
24-hour unattended run: model serves requests, accepts/rejects its own generations, updates overnight, serves again next day. Measure:
- PPL on held-out general set (should not drift more than 5%).
- pass^k on user-simulated queries (should improve day-over-day).
- FF(A) equivalent for the self-improvement phase (should stay near 1.10×).

---

## What we explicitly do NOT do

- **No new CL methods.** SLAO is the shipped core. Self-improvement builds on top, not in parallel.
- **No server-side training.** The whole point is on-device. If something requires a GPU server, it's a bootstrap-only step, not the loop.
- **No growing memory.** LoRA r=32 stays constant. No per-user adapters stacked indefinitely. If we need per-user specialization, it's a separate small adapter merged via SLAO, not a fourth LoRA on the side.
- **No task routing.** The model is one model. It doesn't pick an adapter per input. SLAO's merged state IS the model.

---

## Open questions for the next phase

1. **Is 10k tokens enough bootstrap for SDFT on 350M?** Lin et al. don't specify; we'll measure.
2. **Does the KL anchor hurt plasticity?** AVR fired 0 times in V11 forward order. We may see the same here — useful as a safety net, not a primary mechanism.
3. **What's the pass^k baseline for LFM2.5-350M out of the box?** We don't know. Need to measure before we can claim improvement.
4. **How do we handle conflicting feedback?** User A accepts "X is correct", user B rejects the same generation. Per-user adapters? Or aggregate and accept noise?
5. **What's the right update cadence?** Every accepted sample? Every 100? Overnight batch? Need to measure latency-vs-quality tradeoff.

These get answered in the 6-week plan, not before.
