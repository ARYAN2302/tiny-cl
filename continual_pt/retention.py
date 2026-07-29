"""AVR retention gate: anchor -> verify -> interpolate repair -> commit/reject.

The implementation intentionally operates on held-out capability tests. It does
not claim retention from a loss measured over the same examples used to train.
"""
from __future__ import annotations

import copy

import torch

from .evaluate import evaluate
from .schema import EvalCase, EvalReport


def get_lora_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def set_lora_state(model, state: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if name in state:
            parameter.data.copy_(state[name].to(parameter.device, dtype=parameter.dtype))


def repair_toward_anchor(model, anchor: dict[str, torch.Tensor], alpha: float) -> int:
    """The AVR interpolation: theta <- (1-alpha) theta + alpha theta_anchor."""
    touched = 0
    for name, parameter in model.named_parameters():
        if name in anchor:
            parameter.data.mul_(1.0 - alpha).add_(anchor[name].to(parameter.device, parameter.dtype), alpha=alpha)
            touched += 1
    return touched


class AVRRetentionGate:
    def __init__(self, tolerance: float = 0.0, repair_alpha: float = 0.10, max_repairs: int = 10):
        self.tolerance = tolerance
        self.repair_alpha = repair_alpha
        self.max_repairs = max_repairs

    def baseline(self, model, tokenizer, retained_cases: list[EvalCase]) -> EvalReport:
        return evaluate(model, tokenizer, retained_cases, "retention-baseline")

    def commit_or_rollback(
        self,
        model,
        tokenizer,
        anchor: dict[str, torch.Tensor],
        retention_baseline: EvalReport,
        retained_cases: list[EvalCase],
        target_cases: list[EvalCase],
        target_baseline: EvalReport,
    ) -> tuple[bool, dict]:
        """Keep the candidate only if target skill improves and retention survives.

        AVR repair is only meaningful when a candidate improves the target but
        harms retention. A candidate with no target gain is rejected directly:
        interpolating it toward the anchor cannot turn an unhelpful update into
        evidence of learning, and would waste full evaluation passes.
        """
        log = {"repairs": [], "accepted": False}
        for repair_step in range(self.max_repairs + 1):
            target = evaluate(model, tokenizer, target_cases, f"target-candidate-{repair_step}")
            retention = evaluate(model, tokenizer, retained_cases, f"retention-candidate-{repair_step}")
            target_improved = target.score > target_baseline.score
            retention_ok = retention.score >= retention_baseline.score - self.tolerance
            log["repairs"].append(
                {
                    "step": repair_step,
                    "target": target.to_dict(),
                    "retention": retention.to_dict(),
                    "target_improved": target_improved,
                    "retention_ok": retention_ok,
                }
            )
            if target_improved and retention_ok:
                log["accepted"] = True
                return True, log
            if not target_improved:
                set_lora_state(model, anchor)
                log["rollback"] = "candidate produced no measured target gain"
                return False, log
            if repair_step < self.max_repairs:
                repair_toward_anchor(model, anchor, self.repair_alpha)

        set_lora_state(model, anchor)
        log["rollback"] = "candidate did not improve target while preserving retained capabilities"
        return False, log
