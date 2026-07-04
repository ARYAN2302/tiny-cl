"""
avr.framework — the three-phase continual post-training framework.

The core abstraction: LEARN → VERIFY → REPAIR, each phase pluggable.

    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  LEARN  │────▶│ VERIFY  │────▶│ REPAIR  │
    │ (train) │     │ (drift) │     │ (rewind)│
    └─────────┘     └─────────┘     └─────────┘
         │                                 │
         └─────────── ◀───────────────────┘
                       (if drifted)

Phase 1 (LEARN): fine-tune on the new task. Any training method.
Phase 2 (VERIFY): detect drift on prior tasks. Any drift signal.
Phase 3 (REPAIR): correct the drift. Any weight-space operator.

AVR v1 is one instance:
    LEARN   = SFT (LoRA)
    VERIFY  = PPL-ratio (threshold 1.15)
    REPAIR  = snapshot interpolation (alpha=0.1)

The framework also reserves two future-facing interfaces (v2 research):
    Oracle       — external verifier for grounding (default: noop)
    Consolidator — promote working state to long-term (default: noop)

These are stubs today. They exist in the API so v2 work slots in
without breaking anything.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Callable
import copy
import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────────────
# Data structures
# ────────────────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    """One task in a continual stream."""
    name: str
    train_pairs: List[tuple]        # [(prompt, answer), ...]
    eval_pairs: List[tuple]         # [(prompt, answer), ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftInfo:
    """Output of VERIFY. Says which tasks drifted and how much."""
    drifted_tasks: List[str]
    per_task: Dict[str, Dict[str, float]]   # {task: {ratio, current, best}}
    raw_ppls: Dict[str, float]              # {task: ppl}


@dataclass
class VerificationResult:
    """Output of an Oracle (v2). Today: always 'unverified'."""
    verdict: str            # "verified" | "rejected" | "unverified"
    confidence: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamState:
    """Mutable state carried across the stream."""
    task_index: int = 0                        # 0-indexed position in stream
    completed_tasks: List[str] = field(default_factory=list)
    best_ppls: Dict[str, float] = field(default_factory=dict)
    snapshot: Optional[Dict[str, torch.Tensor]] = None    # last known-good LoRA state
    repair_log: List[Dict[str, Any]] = field(default_factory=list)
    total_repair_steps: int = 0


# ────────────────────────────────────────────────────────────────────
# Pluggable interfaces — the three phases
# ────────────────────────────────────────────────────────────────────

class LearnStrategy(ABC):
    """Phase 1: train on the new task. Any training method goes here."""

    @abstractmethod
    def train(self, model: nn.Module, task: TaskSpec, tokenizer) -> None:
        """Fine-tune `model` on `task`. Mutates model in place."""
        ...


class DriftDetector(ABC):
    """Phase 2: detect drift on prior tasks after LEARN fires."""

    @abstractmethod
    def check(self, model: nn.Module, tokenizer, state: StreamState,
              tasks: List[TaskSpec]) -> DriftInfo:
        """Return which tasks drifted and by how much."""
        ...


class RepairOperator(ABC):
    """Phase 3: correct the drift in weight space.

    v1: SnapshotInterp — global closed-form interpolation.
    v2: SubspaceSnapshotInterp — load-bearing-subspace repair (research).

    The operator does ONE repair step per call. The framework handles
    the verify-repair loop (re-check PPL after each step, stop when
    drift resolves). This matches v23's behavior exactly.
    """

    @abstractmethod
    def repair_step(self, model: nn.Module, snapshot: Dict[str, torch.Tensor],
                    alpha: float) -> int:
        """Apply ONE repair step. Return number of params adjusted."""
        ...

    def repair(self, model: nn.Module, state: StreamState,
               drift: DriftInfo, max_steps: int = 100) -> int:
        """Default repair loop: step + re-check is handled by the framework.
        Override only if you need custom loop logic."""
        # This is overridden by the framework's run_stream which does
        # the verify-repair loop. Kept for backwards compatibility.
        return 1


# ────────────────────────────────────────────────────────────────────
# Future-facing stubs (v2 research) — noop today, slots in later
# ────────────────────────────────────────────────────────────────────

class Oracle(ABC):
    """External verifier. Default: NoopOracle (returns 'unverified').

    v2 will implement:
        ExecOracle   — run code, check it executes and matches expected output
        SympyOracle  — symbolic verification of mathematical claims
        RouterOracle — pick the right oracle per task type

    Today: noop. The framework calls oracle.verify() after LEARN but the
    result is ignored by the default strategy. The interface is here so
    v2 work doesn't break the API.
    """

    @abstractmethod
    def verify(self, model: nn.Module, tokenizer, task: TaskSpec,
               response: str) -> VerificationResult:
        ...


class Consolidator(ABC):
    """Promote from working tier to long-term. Default: NoopConsolidator.

    v2 will implement:
        VerifyGatedConsolidator — only promote after Oracle verification
        FrequencyConsolidator   — promote after N successful uses

    Today: noop. The framework calls consolidator.should_promote() and
    consolidator.promote() after REPAIR, but the defaults do nothing.
    The interface is here so v2 work doesn't break the API.
    """

    @abstractmethod
    def should_promote(self, state: StreamState) -> bool:
        ...

    @abstractmethod
    def promote(self, model: nn.Module, state: StreamState) -> None:
        ...


# ────────────────────────────────────────────────────────────────────
# Noop defaults
# ────────────────────────────────────────────────────────────────────

class NoopOracle(Oracle):
    """Default. Does nothing. v2 swaps in real oracles."""
    def verify(self, model, tokenizer, task, response) -> VerificationResult:
        return VerificationResult(verdict="unverified", confidence=0.0)


class NoopConsolidator(Consolidator):
    """Default. Does nothing. v2 swaps in real consolidators."""
    def should_promote(self, state: StreamState) -> bool:
        return False

    def promote(self, model, state: StreamState) -> None:
        pass


# ────────────────────────────────────────────────────────────────────
# Snapshot helpers — shared across all operators
# ────────────────────────────────────────────────────────────────────

def get_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot all LoRA params to CPU. Constant memory: same size as adapter."""
    return {n: p.data.cpu().clone()
            for n, p in model.named_parameters() if "lora_" in n}


def set_lora_state(model: nn.Module, state: Dict[str, torch.Tensor],
                   device: str = "cuda") -> None:
    """Restore LoRA params from a snapshot."""
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device).to(p.data.dtype))


# ────────────────────────────────────────────────────────────────────
# The orchestrator
# ────────────────────────────────────────────────────────────────────

class ContinualPostTrainer:
    """Runs the LEARN → VERIFY → REPAIR loop over a task stream.

    This is the framework. AVR is one configuration of it:
        learn   = SFT
        verify  = PPLRatioDetector
        repair  = SnapshotInterp
        oracle  = NoopOracle
        consolidator = NoopConsolidator

    v2 configurations:
        learn   = SFT / DPO / GRPO
        verify  = PPLRatio / KL / Hessian / entropy
        repair  = SnapshotInterp / SubspaceSnapshotInterp
        oracle  = ExecOracle / SympyOracle / RouterOracle
        consolidator = VerifyGatedConsolidator

    The trainer doesn't know what each component does. It just calls them
    in order.
    """

    def __init__(self,
                 learn: LearnStrategy,
                 verify: DriftDetector,
                 repair: RepairOperator,
                 oracle: Oracle = None,
                 consolidator: Consolidator = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.learn = learn
        self.verify = verify
        self.repair = repair
        self.oracle = oracle or NoopOracle()
        self.consolidator = consolidator or NoopConsolidator()
        self.device = device

    def run_stream(self, model: nn.Module, tokenizer,
                   tasks: List[TaskSpec],
                   on_task_complete: Optional[Callable] = None) -> StreamState:
        """Train on `tasks` in order. Apply VERIFY+REPAIR between tasks.

        Args:
            model: PEFT-wrapped model with LoRA adapters.
            tokenizer: HF tokenizer.
            tasks: ordered list of TaskSpec.
            on_task_complete: callback(model, tokenizer, state, task_index)
                              called after each task — use for R-matrix eval.

        Returns:
            StreamState with snapshot, repair log, best PPLs.
        """
        state = StreamState()

        for i, task in enumerate(tasks):
            state.task_index = i
            print(f"\n{'='*60}")
            print(f"  Task {i+1}/{len(tasks)}: {task.name}")
            print(f"{'='*60}")

            # ── Phase 1: LEARN ──
            self.learn.train(model, task, tokenizer)

            # ── Phase 2: VERIFY (only if we have prior tasks) ──
            if state.completed_tasks:
                drift = self.verify.check(model, tokenizer, state, tasks)

                if drift.drifted_tasks:
                    print(f"  [VERIFY] Drift on {drift.drifted_tasks}")
                    for t, info in drift.per_task.items():
                        print(f"    {t}: PPL={info['current']:.2f} / "
                              f"best={info['best']:.2f} = {info['ratio']:.2f}x")

                    # ── Phase 3: REPAIR (verify-repair loop, matches v23) ──
                    # Get effective alpha from the repair operator
                    if hasattr(self.repair, '_effective_alpha'):
                        alpha_eff = self.repair._effective_alpha(state.task_index)
                    else:
                        alpha_eff = getattr(self.repair, 'alpha', 0.1)

                    # Max steps from the repair operator (default 10 — v23's shipped cap)
                    max_steps = getattr(self.repair, 'max_steps', 10)

                    n_steps = 0
                    still_drifted = drift
                    for step in range(max_steps):
                        self.repair.repair_step(model, state.snapshot, alpha_eff)
                        n_steps += 1

                        # Re-verify after each step (matches v23's loop)
                        still_drifted = self.verify.check(
                            model, tokenizer, state, tasks)
                        if not still_drifted.drifted_tasks:
                            print(f"  [REPAIR] Converged at step {step+1}")
                            break

                    if still_drifted.drifted_tasks:
                        print(f"  [REPAIR] Max steps ({max_steps}) reached, "
                              f"drift remains on {still_drifted.drifted_tasks}")

                    state.total_repair_steps += n_steps
                    state.repair_log.append({
                        "task": task.name, "repair_steps": n_steps,
                        "drifted": drift.drifted_tasks
                    })
                    print(f"  [REPAIR] {n_steps} steps applied")
                else:
                    print(f"  [VERIFY] No drift")
                    state.repair_log.append({
                        "task": task.name, "repair_steps": 0, "drifted": []
                    })
            else:
                state.repair_log.append({
                    "task": task.name, "repair_steps": 0, "drifted": []
                })

            # ── Update best PPLs (re-eval after repair) ──
            post_ppls = self.verify.check(
                model, tokenizer, state, tasks).raw_ppls
            for t_name, ppl in post_ppls.items():
                if t_name not in state.best_ppls or ppl < state.best_ppls[t_name]:
                    state.best_ppls[t_name] = ppl

            # ── Snapshot for next task's repair target ──
            state.snapshot = get_lora_state(model)

            # ── Oracle (v2 noop) ──
            # Future: verify the model's outputs on this task.
            # Today: returns "unverified", ignored.

            # ── Consolidator (v2 noop) ──
            if self.consolidator.should_promote(state):
                self.consolidator.promote(model, state)

            state.completed_tasks.append(task.name)

            # ── R-matrix callback ──
            if on_task_complete is not None:
                on_task_complete(model, tokenizer, state, i)

        return state
