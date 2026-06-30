"""
Configuration for Tiny-CL experiments.
All hyperparameters and model configs in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ──────────────────────────────────────────────
# Model configurations
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    n_layer: int
    n_head: int
    n_embd: int
    n_ff: int
    vocab_size: int = 50257  # GPT-2 tokenizer
    context_length: int = 256
    dropout: float = 0.1

    @property
    def approx_params(self) -> str:
        """Rough parameter count estimate."""
        emb = self.vocab_size * self.n_embd  # token embedding (shared with output)
        pos = self.context_length * self.n_embd
        per_layer = (
            3 * self.n_embd * self.n_embd +  # QKV
            self.n_embd * self.n_embd +       # output proj
            self.n_embd * self.n_ff +         # FF up
            self.n_ff * self.n_embd +         # FF down
            4 * self.n_embd                   # layer norms
        )
        total = emb + pos + self.n_layer * per_layer
        if total > 1e6:
            return f"~{total / 1e6:.1f}M"
        return f"~{total / 1e3:.0f}K"


MODEL_CONFIGS = {
    "30M": ModelConfig(
        name="30M",
        n_layer=6,
        n_head=8,
        n_embd=384,
        n_ff=1536,
    ),
    "15M": ModelConfig(
        name="15M",
        n_layer=4,
        n_head=4,
        n_embd=256,
        n_ff=1024,
    ),
}


# ──────────────────────────────────────────────
# Data configurations
# ──────────────────────────────────────────────

@dataclass
class PhaseConfig:
    name: str
    dataset_name: str          # HuggingFace dataset name
    dataset_config: Optional[str] = None  # Config/subset name
    text_field: str = "text"   # Field containing the text
    max_words: int = 0         # Max words to use (0 = use all available)
    split: str = "train"


PHASE_CONFIGS = {
    "A": PhaseConfig(
        name="TinyStories",
        dataset_name="roneneldan/TinyStories",
        text_field="text",
        max_words=4_000_000,
    ),
    "B": PhaseConfig(
        name="WikiText",
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        text_field="text",
        max_words=3_000_000,
    ),
    "C": PhaseConfig(
        name="AGNews",
        dataset_name="fancyzhx/ag_news",
        text_field="text",
        max_words=3_000_000,
    ),
}


# ──────────────────────────────────────────────
# Training configurations
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    learning_rate: float = 3e-4
    batch_size: int = 32
    epochs_per_phase: int = 5
    context_length: int = 256
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_samples: int = 2048       # Number of samples for evaluation
    save_dir: str = "checkpoints"
    results_dir: str = "results"
    data_dir: str = "data"
    debug: bool = False            # If True, use tiny data subset


# ──────────────────────────────────────────────
# Method configurations
# ──────────────────────────────────────────────

@dataclass
class MethodConfig:
    name: str
    display_name: str

    # Freeze method
    n_freeze_layers: int = 3

    # Replay method
    replay_buffer_pct: float = 0.01   # Store 1% of old data
    replay_mix_ratio: float = 0.25    # 25% of each batch is replay data

    # Anchor-AVR method
    n_anchor_probes: int = 200        # Number of probe sequences for anchors
    anchor_loss_weight: float = 1.0   # Lambda for anchor-pull loss
    anchor_layers: Optional[List[int]] = None  # Which layers to anchor (None = all)
    anchor_freq: int = 5               # Compute anchor loss every N steps (continuous mode)
    verify_freq: int = 100            # Verify every N steps (discrete mode)
    drift_threshold: float = 0.1      # MSE threshold for triggering repair
    repair_steps: int = 50            # Steps of targeted anchor-pull repair
    repair_lr: float = 1e-4           # Learning rate for repair phase


METHOD_CONFIGS = {
    "naive": MethodConfig(
        name="naive",
        display_name="Naive SGD",
    ),
    "freeze": MethodConfig(
        name="freeze",
        display_name="Freeze Bottom Layers",
        n_freeze_layers=3,
    ),
    "replay": MethodConfig(
        name="replay",
        display_name="Blind Replay 1%",
        replay_buffer_pct=0.01,
        replay_mix_ratio=0.25,
    ),
    "anchor_cont": MethodConfig(
        name="anchor_cont",
        display_name="Anchor-AVR (Continuous)",
        n_anchor_probes=50,
        anchor_loss_weight=1.0,
        anchor_freq=10,
    ),
    "anchor_disc": MethodConfig(
        name="anchor_disc",
        display_name="Anchor-AVR (Discrete)",
        n_anchor_probes=50,
        anchor_loss_weight=1.0,
        anchor_freq=10,
        verify_freq=100,
        drift_threshold=0.1,
        repair_steps=50,
        repair_lr=1e-4,
    ),
}


# ──────────────────────────────────────────────
# Experiment configurations
# ──────────────────────────────────────────────

# Required experiments (do these first)
REQUIRED_EXPERIMENTS = [
    ("30M", "naive"),
    ("30M", "freeze"),
    ("30M", "replay"),
    ("30M", "anchor_cont"),
    ("30M", "anchor_disc"),
]

# Nice-to-have (if budget allows)
OPTIONAL_EXPERIMENTS = [
    ("15M", "naive"),
    ("15M", "replay"),
    ("15M", "anchor_cont"),
]
