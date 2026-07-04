"""
avr.operators — REPAIR phase implementations.

v1 (shipped):
    SnapshotInterp — global closed-form interpolation. Validated in v23.

v2 (research, stub):
    SubspaceSnapshotInterp — repair only the load-bearing subspace.
    Uses probe-gradient SVD to identify which directions to repair.
    The interface is here so it slots in without API changes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn

from .framework import (
    RepairOperator, StreamState, DriftInfo,
    get_lora_state, set_lora_state,
)


class SnapshotInterp(RepairOperator):
    """AVR v1 repair: global closed-form weight interpolation.

        θ_repaired = (1 - α) · θ_current + α · θ_snapshot

    Applied uniformly to every LoRA parameter. Runs in a loop until drift
    is resolved or max_steps is hit.

    Adaptive α: alpha decays with stream position to reduce over-repair
    on later tasks where the snapshot bank is more crowded.
        alpha_effective = alpha / sqrt(task_index + 1)

    Adaptive max_steps: cap based on measured drift ratio, so we don't
    over-repair small drifts.
        max_steps_effective = ceil(log(ratio) / log(1 / (1 - alpha)))
    """

    def __init__(self,
                 alpha: float = 0.1,
                 alpha_decay: str = "sqrt",       # "sqrt" | "none"
                 max_steps: int = 100,
                 max_steps_mode: str = "adaptive",  # "adaptive" | "fixed"
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.max_steps = max_steps
        self.max_steps_mode = max_steps_mode
        self.device = device

    def _effective_alpha(self, task_index: int) -> float:
        if self.alpha_decay == "sqrt":
            return self.alpha / math.sqrt(task_index + 1)
        return self.alpha

    def _effective_max_steps(self, drift: DriftInfo) -> int:
        if self.max_steps_mode != "adaptive" or not drift.per_task:
            return self.max_steps
        # worst-case ratio across drifted tasks
        worst_ratio = max(info["ratio"] for info in drift.per_task.values())
        if worst_ratio <= 1.0:
            return 1
        # how many steps to undo the drift: log(ratio) / log(1/(1-α))
        alpha_eff = self._effective_alpha(drift.per_task and
                                          list(drift.per_task.values())[0].get("_task_index", 0))
        # fallback to fixed if alpha is 0
        if alpha_eff <= 0:
            return self.max_steps
        needed = math.ceil(math.log(worst_ratio) / math.log(1.0 / (1.0 - alpha_eff)))
        return min(self.max_steps, max(1, needed))

    def repair_step(self, model: nn.Module,
                     snapshot: Dict[str, torch.Tensor],
                     alpha: float) -> int:
        """ONE closed-form interpolation step. Returns n params adjusted."""
        n_adj = 0
        for n, p in model.named_parameters():
            if "lora_" in n and n in snapshot:
                snap_val = snapshot[n].to(self.device)
                p.data.copy_((1.0 - alpha) * p.data + alpha * snap_val)
                n_adj += 1
        return n_adj

    def repair(self, model: nn.Module, state: StreamState,
               drift: DriftInfo) -> int:
        """Legacy interface — framework now calls repair_step in a loop.
        Kept for backwards compatibility."""
        if state.snapshot is None:
            return 0
        alpha_eff = self._effective_alpha(state.task_index)
        self.repair_step(model, state.snapshot, alpha_eff)
        return 1


class SubspaceSnapshotInterp(RepairOperator):
    """AVR v2 repair: load-bearing-subspace repair (RESEARCH STUB).

    The idea:
        Δθ = θ_new - θ_snapshot              # what SFT just learned
        g_probe = ∇_θ PPL_probe              # which directions the probe cares about
        Δθ_load = proj(Δθ → span(g_probe))   # component that hurts the probe
        Δθ_free = Δθ - Δθ_load               # harmless component
        θ_repaired = θ_new - α · Δθ_load     # repair only the load-bearing part

    Implementation plan (v2 research, NOT in v1):
        1. Compute probe gradients on old-task eval data (~100 samples)
        2. SVD per LoRA layer (52 tiny SVDs, fast on T4)
        3. Take top-r right singular vectors as load-bearing basis
        4. Project Δθ onto this basis per layer
        5. Repair only the projected component

    This stub raises NotImplementedError. It exists to:
        (a) lock in the interface so v2 doesn't break the API
        (b) document the plan in the codebase
        (c) let users see "subspace_snapshot_interp" as a config option
            that's coming, even though it's not wired yet

    Config:
        repair:
          operator: subspace_snapshot_interp   # selects this class
          rank: 32                              # subspace dimensionality
          n_probe_samples: 100                  # samples for gradient estimate
    """

    def __init__(self,
                 rank: int = 32,
                 n_probe_samples: int = 100,
                 alpha: float = 0.1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.rank = rank
        self.n_probe_samples = n_probe_samples
        self.alpha = alpha
        self.device = device

    def repair_step(self, model: nn.Module,
                     snapshot: Dict[str, torch.Tensor],
                     alpha: float) -> int:
        raise NotImplementedError(
            "SubspaceSnapshotInterp is v2 research. Use SnapshotInterp for v1."
        )

    def repair(self, model: nn.Module, state: StreamState,
               drift: DriftInfo) -> int:
        raise NotImplementedError(
            "SubspaceSnapshotInterp is v2 research. Use SnapshotInterp for v1. "
            "See avr/operators.py for the implementation plan."
        )

    # The actual implementation (v2) will look like:
    #
    # def _compute_load_bearing_subspace(self, model, probe_data, tokenizer):
    #     """SVD on probe gradients per LoRA layer."""
    #     grads_per_layer = {}
    #     for sample in probe_data[:self.n_probe_samples]:
    #         loss = compute_probe_loss(model, sample)
    #         loss.backward(retain_graph=True)
    #         for n, p in model.named_parameters():
    #             if "lora_" in n:
    #                 grads_per_layer.setdefault(n, []).append(
    #                     p.grad.data.clone().cpu().flatten())
    #         model.zero_grad()
    #     # SVD per layer
    #     basis = {}
    #     for n, grad_list in grads_per_layer.items():
    #         G = torch.stack(grad_list)  # (K, P)
    #         U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    #         basis[n] = Vh[:self.rank].T  # (P, r) — load-bearing directions
    #     return basis
    #
    # def repair(self, model, state, drift):
    #     basis = self._compute_load_bearing_subspace(...)
    #     for n, p in model.named_parameters():
    #         if "lora_" not in n or n not in state.snapshot:
    #             continue
    #         delta = p.data.cpu() - state.snapshot[n]   # Δθ
    #         V = basis[n]                                 # (P, r)
    #         delta_load = V @ (V.T @ delta.flatten())     # project
    #         delta_load = delta_load.reshape(p.data.shape)
    #         p.data.copy_(p.data - self.alpha * delta_load.to(self.device))
    #     return 1


# ────────────────────────────────────────────────────────────────────
# Registry — config strings → classes
# ────────────────────────────────────────────────────────────────────

OPERATORS = {
    "snapshot_interp": SnapshotInterp,
    "subspace_snapshot_interp": SubspaceSnapshotInterp,
}


def get_operator(name: str, **kwargs) -> RepairOperator:
    if name not in OPERATORS:
        raise ValueError(f"Unknown repair operator: {name}. "
                         f"Available: {list(OPERATORS.keys())}")
    return OPERATORS[name](**kwargs)
