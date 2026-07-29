# Research Direction: A Continual Post-Training Learner

## Goal

Turn a pretrained language model into a learner that can receive `learn X`,
research its environment, practice, judge whether it improved, update itself,
retain old skills, and reuse what it learned.

This is not a static fine-tuning pipeline. It is a controlled learning process:

```text
goal -> research actions -> source-backed claims -> practice attempt
     -> verifier/value signal -> candidate fast-weight update
     -> AVR retention gate -> slow consolidation -> next learning gap
```

## What current research supports

### 1. Research must be active, not a fixed document list

WebRL shows that online curricula generated from unsuccessful attempts plus
outcome judgment can sustain improvement in web agents. WebAgent-R1 likewise
trains from multi-turn online interaction trajectories rather than static web
text. This motivates a learner-controlled search policy: the model proposes
queries and selects sources, while the framework bounds the available tools.

Sources:

- https://arxiv.org/abs/2411.02337
- https://arxiv.org/abs/2505.16421

### 2. Raw web text is not a learning signal

The failed first experiment demonstrated this locally: document-level SFT
lowered target performance. Learning should be from a source-grounded practice
episode: question, model attempt, externally supported correction, and outcome.
SCoRe finds that offline SFT on idealized self-generated corrections can fail
because it does not match the model's own error distribution. SEAL instead uses
downstream improvement as the signal for learning effective self-edits.

Sources:

- https://arxiv.org/abs/2409.12917
- https://arxiv.org/abs/2506.10943

### 3. Use two speeds of learning

The complementary-learning-systems view separates rapid acquisition of an
episode from slow integration of stable structure. For this project:

- fast memory: a temporary, scoped LoRA update plus source/skill ledger;
- slow memory: an adapter checkpoint accepted only after target, transfer, and
  retention checks.

Source:

- https://arxiv.org/abs/2201.12604

### 4. Keep AVR, but make it a gate—not the learner

AVR's anchor/verify/interpolate-repair rule is the correct final authority for
committing an update:

```text
candidate theta -> verify target and retained skills
  -> repair toward anchor if retention regresses
  -> accept only if target improves and retention survives
  -> otherwise restore anchor exactly
```

Sparse/scoped updates should reduce interference before repair is needed.
Recent sparse-memory finetuning evidence supports limiting updates to parameters
activated by the new knowledge.

Source:

- https://arxiv.org/abs/2510.15103

### 5. Learn only from verifiable progress

For code, math, structured data, or browser tasks, use an executor, test suite,
or task-success outcome as the primary verifier. For research knowledge, use
source spans, multi-source agreement, and a separate held-out source/task set.
Absolute Zero demonstrates why verifiable rewards are especially valuable for
an open-ended self-generated curriculum.

Source:

- https://arxiv.org/abs/2505.03335

### 6. Keep source provenance alongside weights

Parametric knowledge is compressed and lossy. The framework must retain a
source ledger with URL, content hash, quoted support, confidence, and the
adapter/update that learned from it. Source-aware training provides a direct
recipe for associating learned knowledge with sources.

Source:

- https://arxiv.org/abs/2404.01019

### 7. The web is an adversarial environment

Web text has data authority only. It never gains instruction, tool, credential,
or update authority. Browsing runs with an allowlist, no authenticated session,
no arbitrary code execution, and explicit provenance checks. This is required
because indirect prompt injection is effective against existing web agents.

Source:

- https://arxiv.org/abs/2506.07153

## Decisions for v0.2

1. The model plans queries and selects sources; the framework executes only
   search/fetch operations inside trust boundaries.
2. A practice item is retained only when it has exact source evidence and the
   model's unaided attempt shows a measurable gap.
3. Updates are tested as independent, low-learning-rate candidates with narrow
   adapter scopes before an all-LoRA update is attempted.
4. AVR decides whether a candidate becomes durable.
5. Evaluation is established before research begins and split into target,
   retention, and eventually transfer suites.
6. The next implementation milestone is a verifier interface:
   `source_span`, `unit_test`, `structured_output`, and `browser_outcome`.
   The verifier—not the model—must determine whether a practice trajectory is
   successful.

## Findings from the first end-to-end run

The first FSDP2/DTensor run finished without forgetting (retention stayed at
0.875) but did not improve its held-out target score (0.500). This was useful
negative evidence: AVR correctly rejected every update, but the learning
curriculum had two structural faults.

1. Flattened search results let early queries consume the source budget, and
   sequential section traversal let the first FSDP tutorial consume all twelve
   practice slots. The learner never received balanced practice over the goal.
2. The trained targets were offline synthetic answers rather than corrections
   to errors the current learner actually made. SCoRe identifies this exact
   distribution-mismatch failure mode for offline correction SFT.
3. Twelve examples with gradient accumulation of four and one epoch produced
   only three optimizer steps. That was not a meaningful adaptation trial.

The runtime now round-robins search hits and source sections, creates a
source-supported correction after an unaided failed attempt, and evaluates
larger candidate self-edits from the same AVR anchor. This is still a bounded
SFT approximation, not yet full online RL; downstream held-out gain remains
the only commit reward, following SEAL's central design principle.

## Open research questions

- How should an update controller be meta-trained across many learning goals so
  it can choose its own adapter scope, learning rate, and update budget?
- Which abstractions belong in slow weights rather than the source/skill ledger?
- How should the system detect a genuinely new skill versus a distribution shift
  that deserves a separate adapter or expert?
- Can a learned process reward/value model predict whether a candidate update is
  worthwhile before expensive training and evaluation?
- Which transfer tests best distinguish memorization from reusable learning?
