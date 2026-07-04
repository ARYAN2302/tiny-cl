"""
avr.detectors — VERIFY phase implementations.

v1 (shipped):
    PPLRatioDetector — fire if PPL_now / PPL_best > threshold.

v2 (research, future):
    KLDetector     — KL divergence on activations
    HessianDetector — Hessian trace
    EntropyDetector — output entropy

The interface is here; v2 detectors slot in without API changes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn

from .framework import DriftDetector, DriftInfo, StreamState, TaskSpec


def compute_ppl(model: nn.Module, tokenizer, pairs: List[tuple],
                max_samples: int = 200,
                device: str = "cuda") -> float:
    """Compute perplexity on (prompt, answer) pairs."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for prompt, answer in pairs[:max_samples]:
        text = prompt + " " + answer + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, labels=inputs["input_ids"])
        total_loss += out.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def eval_all_ppls(model: nn.Module, tokenizer, tasks: List[TaskSpec],
                  trained_so_far: int, max_samples: int = 200,
                  device: str = "cuda") -> Dict[str, float]:
    """PPL for each task seen so far."""
    ppls = {}
    for i, task in enumerate(tasks):
        if i >= trained_so_far:
            break
        ppls[task.name] = compute_ppl(model, tokenizer, task.eval_pairs,
                                      max_samples, device)
    return ppls


class PPLRatioDetector(DriftDetector):
    """AVR v1 detector: fire if PPL_now / PPL_best > threshold.

    Two threshold modes:
        fixed:    threshold = 1.15 (the v23 default)
        adaptive: threshold = mu + k*sigma of running PPL distribution
                  (v2 principled default — not yet wired, falls back to fixed)
    """

    def __init__(self,
                 threshold: float = 1.15,
                 threshold_mode: str = "fixed",  # "fixed" | "adaptive"
                 adaptive_k: float = 2.0,        # k for mu + k*sigma
                 probe_samples: int = 200,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.adaptive_k = adaptive_k
        self.probe_samples = probe_samples
        self.device = device

    def _effective_threshold(self, state: StreamState) -> float:
        if self.threshold_mode != "adaptive":
            return self.threshold
        # Adaptive: threshold = mu + k*sigma of best PPLs seen so far.
        # If we don't have enough data points, fall back to fixed.
        if len(state.best_ppls) < 3:
            return self.threshold
        ppls = list(state.best_ppls.values())
        mu = sum(ppls) / len(ppls)
        var = sum((p - mu) ** 2 for p in ppls) / len(ppls)
        sigma = math.sqrt(var) if var > 0 else 0.0
        # adaptive threshold = mu * (1 + k * sigma / mu)  — multiplicative form
        # so it stays a ratio, not an absolute PPL
        cv = sigma / mu if mu > 0 else 0.0
        return 1.0 + self.adaptive_k * cv

    def check(self, model: nn.Module, tokenizer,
              state: StreamState, tasks: List[TaskSpec]) -> DriftInfo:
        """Compute PPL on all prior tasks, flag any above threshold."""
        trained_so_far = state.task_index + 1
        current_ppls = eval_all_ppls(
            model, tokenizer, tasks, trained_so_far,
            self.probe_samples, self.device)

        threshold = self._effective_threshold(state)
        drifted_tasks = []
        per_task = {}

        for task_name in state.completed_tasks:
            if task_name not in current_ppls or task_name not in state.best_ppls:
                continue
            current = current_ppls[task_name]
            best = state.best_ppls[task_name]
            ratio = current / best if best > 0 else 1.0
            if ratio > threshold:
                drifted_tasks.append(task_name)
                per_task[task_name] = {
                    "current": current,
                    "best": best,
                    "ratio": ratio,
                    "threshold": threshold,
                }

        return DriftInfo(
            drifted_tasks=drifted_tasks,
            per_task=per_task,
            raw_ppls=current_ppls,
        )


# v2 stubs — interfaces locked in, implementations come later

class KLDetector(DriftDetector):
    """v2: detect drift via KL divergence on activation distributions.

    Not implemented. Interface reserved.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("KLDetector is v2 research.")

    def check(self, model, tokenizer, state, tasks) -> DriftInfo:
        raise NotImplementedError


class HessianDetector(DriftDetector):
    """v2: detect drift via Hessian trace (loss curvature change).

    Not implemented. Interface reserved.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("HessianDetector is v2 research.")

    def check(self, model, tokenizer, state, tasks) -> DriftInfo:
        raise NotImplementedError


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────

DETECTORS = {
    "ppl_ratio": PPLRatioDetector,
    "kl": KLDetector,
    "hessian": HessianDetector,
}


def get_detector(name: str, **kwargs) -> DriftDetector:
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector: {name}. "
                         f"Available: {list(DETECTORS.keys())}")
    return DETECTORS[name](**kwargs)
