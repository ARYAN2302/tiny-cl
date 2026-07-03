"""
avr.strategy — AVRStrategy, the orchestrator that wires LEARN+VERIFY+REPAIR.

The ContinualPostTrainer in framework.py is generic — it just calls the
three phases in order. AVRStrategy is the AVR-specific configuration:
it builds the right detector, operator, and learn strategy from a config
dict, then runs the trainer.

This is what `avr train config.yaml` calls.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from .framework import (
    ContinualPostTrainer, StreamState, TaskSpec,
    LearnStrategy, get_lora_state,
)
from .detectors import get_detector
from .operators import get_operator
from .trainer import SFTStrategy, ReplaySFTStrategy


LEARN_STRATEGIES = {
    "sft": SFTStrategy,
    "replay_sft": ReplaySFTStrategy,
}


def get_learn_strategy(name: str, **kwargs) -> LearnStrategy:
    if name not in LEARN_STRATEGIES:
        raise ValueError(f"Unknown learn strategy: {name}. "
                         f"Available: {list(LEARN_STRATEGIES.keys())}")
    return LEARN_STRATEGIES[name](**kwargs)


class AVRStrategy:
    """Build the trainer from a config dict and run it.

    Config shape (from YAML):
        model: {id, lora_targets, lora_rank, lora_alpha, lora_dropout}
        stream: {benchmark, tasks, seed}
        learn: {method, epochs, lr, batch_size, context_length}
        verify: {detector, threshold, probe_samples}
        repair: {operator, alpha, alpha_decay, max_steps}
        oracle: noop  (v2)
        consolidator: noop  (v2)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

    def build_trainer(self) -> ContinualPostTrainer:
        learn_cfg = self.config.get("learn", {})
        verify_cfg = self.config.get("verify", {})
        repair_cfg = self.config.get("repair", {})

        learn = get_learn_strategy(
            learn_cfg.get("method", "sft"),
            epochs=learn_cfg.get("epochs", 3),
            lr=learn_cfg.get("lr", 2e-4),
            weight_decay=learn_cfg.get("weight_decay", 0.01),
            max_grad_norm=learn_cfg.get("max_grad_norm", 1.0),
            batch_size=learn_cfg.get("batch_size", 8),
            context_length=learn_cfg.get("context_length", 512),
            device=self.device,
        )

        verify = get_detector(
            verify_cfg.get("detector", "ppl_ratio"),
            threshold=verify_cfg.get("threshold", 1.15),
            threshold_mode=verify_cfg.get("threshold_mode", "fixed"),
            probe_samples=verify_cfg.get("probe_samples", 200),
            device=self.device,
        )

        repair = get_operator(
            repair_cfg.get("operator", "snapshot_interp"),
            alpha=repair_cfg.get("alpha", 0.1),
            alpha_decay=repair_cfg.get("alpha_decay", "sqrt"),
            max_steps=repair_cfg.get("max_steps", 100),
            max_steps_mode=repair_cfg.get("max_steps_mode", "adaptive"),
            device=self.device,
        )

        return ContinualPostTrainer(
            learn=learn, verify=verify, repair=repair,
            device=self.device,
        )

    def run(self, model: nn.Module, tokenizer, tasks: List[TaskSpec],
            on_task_complete=None) -> StreamState:
        trainer = self.build_trainer()
        return trainer.run_stream(model, tokenizer, tasks, on_task_complete)
