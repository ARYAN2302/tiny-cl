"""
V3 Configuration: The Living Model on LFM2.5-350M.
Fast-slow architecture + AVR + consolidation.
All hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ──────────────────────────────────────────────
# LFM2.5 Architecture Constants
# ──────────────────────────────────────────────

# LFM2.5-350M has 16 layers: 10 conv (fast) + 6 attention (slow)
CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]
ALL_LAYER_IDS = CONV_LAYER_IDS + ATTN_LAYER_IDS


# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str
    hf_id: str
    display_name: str
    hidden_dim: int = 1024
    n_conv_layers: int = 10
    n_attn_layers: int = 6
    conv_layer_ids: List[int] = field(default_factory=lambda: CONV_LAYER_IDS.copy())
    attn_layer_ids: List[int] = field(default_factory=lambda: ATTN_LAYER_IDS.copy())
    context_length: int = 512


LFM2_5_CONFIG = ModelConfig(
    name="lfm2.5-350M",
    hf_id="LiquidAI/LFM2.5-350M",
    display_name="LFM2.5-350M",
    hidden_dim=1024,
    n_conv_layers=10,
    n_attn_layers=6,
)

PRETRAINED_MODELS = {
    "lfm2.5-350M": LFM2_5_CONFIG,
}


# ──────────────────────────────────────────────
# Domain (Data) Configurations
# ──────────────────────────────────────────────

@dataclass
class DomainConfig:
    name: str
    display_name: str
    dataset_name: str
    dataset_config: Optional[str] = None
    text_field: str = "text"
    split: str = "train"
    max_tokens: int = 0
    is_safety_critical: bool = False  # Amygdala: medical = True
    salience: float = 1.0            # Default salience weight

HERO_DOMAINS = {
    "A": DomainConfig(
        name="medical",
        display_name="Medical",
        dataset_name="epfl-llm/guidelines",
        text_field="clean_text",
        max_tokens=2_000_000,
        is_safety_critical=True,
        salience=2.0,  # Medical knowledge = high priority
    ),
    "B": DomainConfig(
        name="code",
        display_name="Code",
        dataset_name="iamtarun/python_code_instructions_18k_alpaca",
        text_field="output",
        max_tokens=2_000_000,
        is_safety_critical=False,
        salience=1.0,
    ),
    "C": DomainConfig(
        name="creative",
        display_name="Creative Writing",
        dataset_name="roneneldan/TinyStories",
        text_field="text",
        max_tokens=2_000_000,
        is_safety_critical=False,
        salience=0.5,  # Creative = lowest priority to protect
    ),
}

EXPERIMENT_CONFIGS = {
    "hero": HERO_DOMAINS,
}


# ──────────────────────────────────────────────
# LoRA Configuration (Fast-Slow Split)
# ──────────────────────────────────────────────

@dataclass
class FastSlowLoRAConfig:
    """Separate LoRA configs for fast (conv) and slow (attention) pathways."""

    # Fast path: Conv layers (hippocampus)
    fast_rank: int = 16
    fast_alpha: int = 32
    fast_dropout: float = 0.05
    fast_lr: float = 2e-4        # High LR — rapid absorption
    fast_target_modules: List[str] = field(default_factory=lambda: ["in_proj", "out_proj"])

    # Slow path: Attention layers (neocortex)
    slow_rank: int = 16
    slow_alpha: int = 32
    slow_dropout: float = 0.05
    slow_lr: float = 5e-5        # Low LR — careful integration
    slow_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Shared
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ──────────────────────────────────────────────
# Living Model Configuration
# ──────────────────────────────────────────────

@dataclass
class LivingModelConfig:
    """Configuration for the full Living Model with AVR + consolidation."""

    # Verification Head
    verification_hidden_dim: int = 128   # MLP intermediate size (~75K params)
    verification_threshold: float = 0.85 # Health score below this triggers repair

    # AVR Loop (same as V2, now on structural split)
    n_anchor_probes: int = 50
    drift_threshold: float = 0.1
    repair_steps: int = 150              # More steps — 50 wasn't enough (always "Partial")
    repair_lr: float = 1e-4

    # Consolidation (V3) — disabled for now, net-negative in current form
    # Consolidation modifies attn LoRA which disrupts previous domains faster
    # than it stabilizes them. Re-enable with per-domain attn heads or EWC.
    consolidation_steps: int = 0         # 0 = disabled
    consolidation_lr: float = 1e-4
    consolidation_temp: float = 2.0
    fast_reset_factor: float = 1.0       # No reset

    # Phase Controller
    verify_every_n_steps: int = 100      # Fallback: verify every N steps
    auto_verify_on_domain_change: bool = True  # Auto-verify when domain shifts


# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    learning_rate: float = 2e-4       # Default for fast path
    batch_size: int = 16
    epochs_per_phase: int = 1
    context_length: int = 512
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_samples: int = 2048
    save_dir: str = "v3/checkpoints"
    results_dir: str = "v3/results"
    data_dir: str = "v3/data"
    log_interval: int = 10


# ──────────────────────────────────────────────
# Naive Baseline Configuration (for comparison)
# ──────────────────────────────────────────────

@dataclass
class NaiveLoRAConfig:
    """Standard LoRA on attention only - same as V2 baseline."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
