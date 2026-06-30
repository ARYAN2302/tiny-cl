"""
V4 Config: Streaming AVR experiment.
Two model scales, variable increment sizes, three methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ──────────────────────────────────────────────
# Model Architectures
# ──────────────────────────────────────────────

@dataclass
class LSTMConfig:
    """Gboard-scale CIFG LSTM: 1.4M params, ~1.4MB."""
    vocab_size: int = 10000
    embed_dim: int = 96
    hidden_dim: int = 670
    n_layers: int = 1
    context_length: int = 64      # Short context for next-word prediction
    name: str = "lstm_1.4M"


@dataclass
class SmallGPTConfig:
    """V1-scale GPT: 30M params, trained from scratch."""
    vocab_size: int = 10000
    embed_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 2048
    context_length: int = 256
    name: str = "gpt_30M"


MODEL_CONFIGS = {
    "lstm_1.4M": LSTMConfig(),
    "gpt_30M": SmallGPTConfig(),
}


# ──────────────────────────────────────────────
# Domain Configurations
# ──────────────────────────────────────────────

@dataclass
class DomainConfig:
    name: str
    display_name: str
    dataset_name: str
    text_field: str = "text"
    split: str = "train"
    max_tokens: int = 0  # 0 = use all available


DOMAINS = {
    "A": DomainConfig(
        name="medical",
        display_name="Medical",
        dataset_name="epfl-llm/guidelines",
        text_field="clean_text",
        max_tokens=500_000,  # Smaller for CPU
    ),
    "B": DomainConfig(
        name="code",
        display_name="Code",
        dataset_name="iamtarun/python_code_instructions_18k_alpaca",
        text_field="output",
        max_tokens=500_000,
    ),
}


# ──────────────────────────────────────────────
# Streaming / Increment Configuration
# ──────────────────────────────────────────────

INCREMENT_SIZES = [0, 500, 100, 20]  # 0 = full-phase (baseline)


# ──────────────────────────────────────────────
# Method Configurations
# ──────────────────────────────────────────────

@dataclass
class AVRConfig:
    n_anchor_probes: int = 20       # Fewer probes for streaming regime
    drift_threshold: float = 0.1
    repair_steps: int = 30          # Fewer steps for small increments
    repair_lr: float = 1e-4
    verify_every_n_increments: int = 5  # Check every N increments


@dataclass
class EWCConfig:
    lambda_: float = 0.1
    fisher_n_samples: int = 200     # Needs full pass — will flag when impossible


# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    eval_samples: int = 512
    results_dir: str = "v4/results"
    log_interval: int = 10


# ──────────────────────────────────────────────
# Experiment Grid
# ──────────────────────────────────────────────

METHODS = ["naive", "avr", "ewc"]
