# continual-pt

`continual-pt` is a new post-training runtime for a model that is told to
**learn X**. It researches the web, creates source-grounded learning
experiences, trains temporary fast weights, and commits only updates that
improve the target while an AVR retention gate prevents forgetting.

It is deliberately separate from `tiny-cl/`. AVR is used here as one correct
retention subsystem, not as the product architecture.

For the research record, completed-run outcomes, limitations, and takeover
context, see [AGENT_HANDOFF.md](AGENT_HANDOFF.md).

```text
learn X -> browse trusted sources -> practice + judge -> temporary LoRA update
        -> target evaluation + retained-skill evaluation -> AVR repair/rollback
        -> commit durable update -> learn next gap
```

## T4 target

The default model is `Qwen/Qwen3-4B-Instruct-2507`, loaded in 4-bit NF4 with
LoRA rank 16. This is intentionally sized for a single NVIDIA T4 (16 GB).

## A real learning goal

Goals are supplied as YAML. The evaluations are held out from training and are
run before and after each learning cycle. The model writes its own research
queries, invokes the bounded web-search/fetch tools, and selects which fetched
sources become training material. Trusted domains constrain that tool use.

```yaml
name: learn-fastapi-dependency-injection
objective: Learn to explain and correctly use FastAPI dependency injection.
research_queries:
  - FastAPI dependencies tutorial
sources:
  - https://fastapi.tiangolo.com/tutorial/dependencies/
target_eval:
  - id: dependency-explain
    prompt: In FastAPI, what does Depends do and when should it be used?
    must_contain: ["dependency", "Depends"]
  - id: dependency-code
    prompt: Write a minimal FastAPI route that injects a shared query parameter dependency.
    must_contain: ["Depends", "def"]
retention_eval:
  - id: retained-python
    prompt: What does a Python context manager guarantee around a with block?
    must_contain: ["enter", "exit"]
```

Run locally after installing GPU dependencies:

```bash
pip install -e '.[gpu,modal]'
continual-pt learn goals/fastapi.yaml --cycles 2
```

The output directory contains the baseline, every candidate-update decision,
the final evaluation, sources, grounded examples, and adapter checkpoints.

## Safety boundaries

Web pages are untrusted data. They cannot authorize tools, alter the objective,
or trigger an update by themselves. The learner records source provenance and
requires a quoted source span for generated learning examples. It also uses a
shadow adapter and restores the anchor if AVR cannot recover retention.

## Modal

`continual_pt/modal_app.py` provides a single-T4 Modal entry point. It is
ready to run after Modal authentication is configured; no cloud job is started
by this repository automatically.
