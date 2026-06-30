"""
V2 Configuration: Pretrained models + LoRA + Anchor-AVR.
All hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# ──────────────────────────────────────────────
# Pretrained Model Configurations
# ──────────────────────────────────────────────

@dataclass
class PretrainedModelConfig:
    name: str
    hf_id: str
    display_name: str
    is_standard_transformer: bool = True  # False for LFM2.5 (hybrid arch)
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    context_length: int = 512
    trust_remote_code: bool = False


PRETRAINED_MODELS = {
    "smollm2-360M": PretrainedModelConfig(
        name="smollm2-360M",
        hf_id="HuggingFaceTB/SmolLM2-360M",
        display_name="SmolLM2-360M",
        is_standard_transformer=True,
        lora_target_modules=["q_proj", "v_proj"],
        context_length=512,
        trust_remote_code=False,
    ),
    "smollm2-135M": PretrainedModelConfig(
        name="smollm2-135M",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        display_name="SmolLM2-135M",
        is_standard_transformer=True,
        lora_target_modules=["q_proj", "v_proj"],
        context_length=512,
        trust_remote_code=False,
    ),
    "lfm2.5-350M": PretrainedModelConfig(
        name="lfm2.5-350M",
        hf_id="LiquidAI/LFM2.5-350M",
        display_name="LFM2.5-350M",
        is_standard_transformer=False,
        lora_target_modules=["q_proj", "v_proj"],  # Placeholder — smoke test will determine
        context_length=512,
        trust_remote_code=True,  # Custom model code
    ),
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
    max_tokens: int = 0  # 0 = use all available


# Sanity run: same-domain incremental (WikiText-2 split in half)
SANITY_DOMAINS = {
    "A": DomainConfig(
        name="wikitext_first_half",
        display_name="WikiText (1st half)",
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        text_field="text",
        max_tokens=2_000_000,
    ),
    "B": DomainConfig(
        name="wikitext_second_half",
        display_name="WikiText (2nd half)",
        dataset_name="Salesforce/wikitext",
        dataset_config="wikitext-103-raw-v1",
        text_field="text",
        max_tokens=2_000_000,
    ),
}

# Hero run: max-shift domains
HERO_DOMAINS = {
    "A": DomainConfig(
        name="medical",
        display_name="Medical",
        dataset_name="epfl-llm/guidelines",
        text_field="clean_text",
        max_tokens=2_000_000,
    ),
    "B": DomainConfig(
        name="code",
        display_name="Code",
        dataset_name="iamtarun/python_code_instructions_18k_alpaca",
        text_field="output",
        max_tokens=2_000_000,
    ),
    "C": DomainConfig(
        name="creative",
        display_name="Creative Writing",
        dataset_name="roneneldan/TinyStories",
        text_field="text",
        max_tokens=2_000_000,
    ),
}

# All experiment configs
EXPERIMENT_CONFIGS = {
    "sanity": SANITY_DOMAINS,
    "hero": HERO_DOMAINS,
}


# ──────────────────────────────────────────────
# LoRA Configuration
# ──────────────────────────────────────────────

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32       # 2x rank (standard practice)
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # Override from model config if None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


# ──────────────────────────────────────────────
# Training Configuration
# ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    learning_rate: float = 2e-4
    batch_size: int = 16
    epochs_per_phase: int = 1
    context_length: int = 512
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_samples: int = 2048
    save_dir: str = "v2/checkpoints"
    results_dir: str = "v2/results"
    data_dir: str = "v2/data"
    debug: bool = False  # If True, use tiny data subset
    log_interval: int = 10


# ──────────────────────────────────────────────
# Method Configurations
# ──────────────────────────────────────────────

@dataclass
class MethodConfig:
    name: str
    display_name: str

    # Replay method
    replay_buffer_pct: float = 0.01
    replay_mix_ratio: float = 0.25

    # EWC method
    ewc_lambda: float = 0.1
    ewc_fisher_n_samples: int = 200

    # Anchor-AVR method
    n_anchor_probes: int = 50
    anchor_loss_weight: float = 1.0
    anchor_layers: Optional[List[int]] = None  # None = all layers
    anchor_freq: int = 10          # Continuous mode
    verify_freq: int = 100         # Discrete mode
    drift_threshold: float = 0.1
    repair_steps: int = 50
    repair_lr: float = 1e-4


METHOD_CONFIGS = {
    "naive": MethodConfig(
        name="naive",
        display_name="Naive LoRA",
    ),
    "replay": MethodConfig(
        name="replay",
        display_name="LoRA + Replay 1%",
        replay_buffer_pct=0.01,
        replay_mix_ratio=0.25,
    ),
    "ewc": MethodConfig(
        name="ewc",
        display_name="LoRA + EWC",
        ewc_lambda=0.1,
        ewc_fisher_n_samples=200,
    ),
    "anchor_cont": MethodConfig(
        name="anchor_cont",
        display_name="LoRA + Anchor-AVR (Continuous)",
        n_anchor_probes=50,
        anchor_loss_weight=1.0,
        anchor_freq=10,
    ),
    "anchor_disc": MethodConfig(
        name="anchor_disc",
        display_name="LoRA + Anchor-AVR (Discrete)",
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
# Experiment Definitions
# ──────────────────────────────────────────────

# Sanity run: 2 methods on smollm2-360M with same-domain data
SANITY_EXPERIMENTS = [
    ("smollm2-360M", "naive", "sanity"),
    ("smollm2-360M", "anchor_disc", "sanity"),
]

# Hero run: all 5 methods on smollm2-360M with max-shift data
HERO_EXPERIMENTS = [
    ("smollm2-360M", "naive", "hero"),
    ("smollm2-360M", "ewc", "hero"),
    ("smollm2-360M", "anchor_disc", "hero"),
]

# LFM2.5 moon shot (same 5 methods)
LFM_EXPERIMENTS = [
    ("lfm2.5-350M", "naive", "hero"),
    ("lfm2.5-350M", "anchor_disc", "hero"),
]

# 135M ablation
ABLATION_EXPERIMENTS = [
    ("smollm2-135M", "naive", "hero"),
    ("smollm2-135M", "anchor_cont", "hero"),
    ("smollm2-135M", "anchor_disc", "hero"),
]
