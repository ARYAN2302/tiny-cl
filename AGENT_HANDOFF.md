# Handoff: Continual PT Framework

## Mission

Build a standalone post-training framework that can turn a pretrained model
into a continual learner. The intended loop is:

```text
learn X -> model plans research -> bounded web tools fetch primary sources
        -> model identifies its own knowledge gaps -> source-backed correction
        -> temporary weight update -> target + retention evaluation
        -> AVR repair/rollback or durable commit -> next gap
```

This is **not** an extension of `tiny-cl`, and AVR is not the product. AVR is
the retention subsystem that protects existing skills while a new continual
post-training system learns useful real-world knowledge through weight updates.

## Repository layout

- `continual_pt/`: the runnable framework.
- `goals/pytorch-distributed.yaml`: real initial learning goal on current
  PyTorch FSDP2, DTensor, DeviceMesh, tensor parallelism, and pipeline
  parallelism.
- `continual_pt/modal_app.py`: single-T4 Modal entry point.
- `RESEARCH_DIRECTION.md`: literature-driven design record.
- `tests/`: AVR and balanced-curriculum regression tests.
- `tiny-cl/`: ignored reference clone; do not merge its code into the new
  framework unless explicitly changing scope.

## Architecture today

1. `ResearchAgent` has the model propose web queries and select complementary
   primary sources.
2. `WebResearcher` executes only bounded search/fetch actions against
   allowlisted domains. Web text is data, never executable instruction.
3. `GroundedExampleBuilder` creates a practical question from an exact source
   quote, obtains the model's unaided answer, and retains only a model-generated
   correction that overlaps the quoted evidence. This makes the training data
   reflect a real error the current model made.
4. Candidate LoRA updates begin from the same anchor. The selected update
   schedules differ only in scoped parameter set, learning rate, and budget.
5. `AVRRetentionGate` evaluates held-out target and retention suites. It
   accepts only strict target improvement with sufficient retention, repairs
   only candidates that improve target but harm retention, and otherwise
   restores the exact anchor before the next candidate.

Default runtime: `Qwen/Qwen3-4B-Instruct-2507`, 4-bit NF4, LoRA rank 16, one
Modal T4 (16GB). Model and result caches live in Modal volumes
`continual-pt-hf-cache` and `continual-pt-results`.

## Research that shaped the design

- **SEAL**: a model can generate self-edits, but the edit policy must be
  selected by downstream post-update performance, not its stated confidence.
  https://arxiv.org/abs/2506.10943
- **SCoRe**: offline synthetic correction SFT has distribution mismatch; the
  learner should train from errors produced by its own current policy.
  https://arxiv.org/abs/2409.12917
- **Self-RAG**: retrieval should be coupled to critique/evidence support, not
  blindly appended to a prompt or training corpus.
  https://arxiv.org/abs/2310.11511
- **Continual pre-training at scale**: replay and meaningful LR schedules are
  practical foundations for adaptation, but must be applied without sacrificing
  the held-out retention gate.
  https://arxiv.org/abs/2403.08763
- **WebAgent-R1**: online interaction plus externally verifiable outcomes is
  the right long-term target for real task learning.
  https://arxiv.org/abs/2505.16421

The immediate conclusion is that the next evolution should be a verifier
interface (`source_span`, `unit_test`, `structured_output`, `browser_outcome`)
and eventually an outcome-trained edit controller. Do not replace the
downstream gate with loss reduction or a self-score.

## Run history

All runs used the same held-out initial suite: 8 target cases and 8 unrelated
retention cases. The target cases were not used as training prompts.

| Stage | Result | Interpretation |
| --- | --- | --- |
| Initial Qwen2.5 1.5B run | interrupted by T4 preemption | Added Modal cache persistence and resumable result storage. |
| First 1.5B attempt | target 0.500; candidate reduced target | Raw/weak synthetic SFT was not a valid learning signal. |
| First Qwen3 4B attempt | target 0.500 -> 0.500; retention 0.875 | AVR correctly rejected all candidates. It exposed only three optimizer steps and first-document curriculum starvation. |
| Corrected Qwen3 4B run | target **0.500 -> 0.625**; retention **0.875 -> 1.000** | `balanced-all-lora`, 3 epochs, 10 source-verified correction examples was accepted. The gained held-out case was `pipeline-schedule`; no target case regressed. |

The historical Modal apps are stopped. The latest successful app was
`ap-9Z7plx97P5xzMSrhTi4ijH`; its persisted result is in the
`continual-pt-results` volume under
`learn-pytorch-fsdp2-dtensor-20260729-184319/`.

## What worked

- The model autonomously planned real PyTorch documentation queries.
- Trusted-domain web research and exact evidence spans made training examples
  auditable.
- The model had genuine knowledge gaps before training.
- AVR exact-anchor rollback prevented all rejected candidates from contaminating
  future trials.
- Balanced source coverage plus correction-on-own-attempt produced the first
  accepted weight update and a held-out target gain.
- Persistent Modal model cache avoided repeated base-model downloads after T4
  preemption.

## What did not work, and why

- Training raw source text or lightly checked synthetic answers: no target gain
  and sometimes degradation.
- Flattening search results: early queries monopolized the six-document budget.
- Iterating documents/sections sequentially: all 12 examples came from early
  FSDP material, leaving DeviceMesh, DTensor, TP, and pipeline content unseen.
- One epoch with gradient accumulation four: only three optimizer steps, an
  inadequate experiment rather than evidence of no learnability.
- Applying AVR interpolation to a candidate with no target gain: it wastes
  evaluation time. The gate now rejects that candidate immediately.

## Current limits

- The target suite is only eight cases and the lexical `must_contain` scorer is
  a coarse proxy. A 0.125 gain equals one case; do not call this broad research
  competence.
- The correction verifier is source-overlap based and uses the same model to
  formulate corrections. It reduces hallucinated training material but is not a
  substitute for an external task verifier.
- No replay buffer is yet mixed into accepted updates. AVR evaluates retention,
  but replay should be added once the skill ledger has enough verified examples.
- There is no trained self-edit controller yet; candidate schedules are still
  framework-specified and selected by outcome.

## Next agent priorities

1. Add a typed verifier interface and use real executable outcomes where
   possible. For the PyTorch goal, use syntax/import/static-plan checks first;
   use a multi-GPU executor only where hardware is available.
2. Expand evaluation into held-out target, transfer, and retention suites with
   semantic rubrics or executable tests. Keep all evaluation prompts excluded
   from practice construction.
3. Persist a source/skill ledger containing URL, content hash, evidence span,
   correction, adapter version, and acceptance outcome.
4. Add a small verified replay reservoir from previous accepted skills and
   evaluate replay ratios under the same AVR gate.
5. Train or search an edit controller across many goals only after outcome
   verifiers exist. SEAL-style self-edits should be rewarded by post-update
   success, not trusted by default.

## Operational notes

- Do not commit Modal credentials, Hugging Face tokens, or volume contents.
- Use `python3 -m modal run --detach continual_pt/modal_app.py::run_goal ...`
  for long runs; monitor with `python3 -m modal app logs <app-id>`.
- Before a cloud run, run `python3 -m compileall -q continual_pt tests` and
  `git diff --check`. Local CPU tests require `torch` and `pytest`, which were
  not installed in the development environment used here.
