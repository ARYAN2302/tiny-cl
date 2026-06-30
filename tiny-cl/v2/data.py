"""
V2 Data Pipeline: Download + tokenize + split data for pretrained models.
Supports sanity (same-domain) and hero (max-shift) experiments.
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import DomainConfig, TrainConfig


class TextDataset(Dataset):
    """Tokenized text dataset for causal LM training.
    
    Uses NON-OVERLAPPING chunks (stride = context_length).
    This avoids the sliding-window blowup that creates millions of
    overlapping sequences from a single document.
    """

    def __init__(self, token_ids: torch.Tensor, context_length: int):
        self.context_length = context_length
        # Pre-chunk into non-overlapping sequences
        n_tokens = len(token_ids)
        n_complete = (n_tokens // (context_length + 1)) * (context_length + 1)
        trimmed = token_ids[:n_complete]
        self.chunks = trimmed.reshape(-1, context_length + 1)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def load_and_tokenize_domain(
    domain_config: DomainConfig,
    tokenizer: AutoTokenizer,
    context_length: int = 512,
    max_tokens: int = 0,
    seed: int = 42,
    debug: bool = False,
) -> torch.Tensor:
    """
    Download and tokenize a single domain's data.
    Returns a flat tensor of token IDs.
    """
    from datasets import load_dataset

    print(f"  Loading {domain_config.display_name} from {domain_config.dataset_name}...")

    # Load dataset
    try:
        if domain_config.dataset_config:
            dataset = load_dataset(
                domain_config.dataset_name,
                domain_config.dataset_config,
                split=domain_config.split,
            )
        else:
            dataset = load_dataset(
                domain_config.dataset_name,
                split=domain_config.split,
            )
    except Exception as e:
        print(f"  Warning: Could not load {domain_config.dataset_name}: {e}")
        print(f"  Trying alternative split or config...")
        # Try 'train' split as fallback
        try:
            dataset = load_dataset(
                domain_config.dataset_name,
                split="train",
            )
        except Exception as e2:
            raise RuntimeError(f"Failed to load {domain_config.dataset_name}: {e2}")

    # Extract text field
    texts = dataset[domain_config.text_field]

    # Filter empty texts
    texts = [t for t in texts if t and len(t.strip()) > 20]

    if debug:
        texts = texts[:100]

    print(f"  Raw texts: {len(texts)}")

    # Tokenize
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

        if max_tokens > 0 and len(all_tokens) >= max_tokens:
            all_tokens = all_tokens[:max_tokens]
            break

    token_tensor = torch.tensor(all_tokens, dtype=torch.long)
    print(f"  Tokenized: {len(token_tensor):,} tokens")

    return token_tensor


def prepare_domain_data(
    domain_config: DomainConfig,
    tokenizer: AutoTokenizer,
    train_config: TrainConfig,
    seed: int = 42,
) -> Tuple[TextDataset, TextDataset]:
    """
    Prepare train and validation datasets for a single domain.
    Returns (train_dataset, val_dataset).
    """
    max_tokens = domain_config.max_tokens if domain_config.max_tokens > 0 else 0

    all_tokens = load_and_tokenize_domain(
        domain_config,
        tokenizer,
        context_length=train_config.context_length,
        max_tokens=max_tokens,
        seed=seed,
        debug=train_config.debug,
    )

    # 90/10 train/val split
    n_tokens = len(all_tokens)
    n_val = max(train_config.context_length, int(n_tokens * 0.1))
    n_train = n_tokens - n_val

    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    train_dataset = TextDataset(train_tokens, train_config.context_length)
    val_dataset = TextDataset(val_tokens, train_config.context_length)

    print(f"  Train: {len(train_dataset):,} sequences | Val: {len(val_dataset):,} sequences")

    return train_dataset, val_dataset


def prepare_all_domains(
    domains: Dict[str, DomainConfig],
    tokenizer_name: str,
    train_config: TrainConfig,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Prepare data for all domains in an experiment.
    Returns {phase_key: {"train": TextDataset, "val": TextDataset, "config": DomainConfig}}
    """
    print(f"\n{'='*60}")
    print(f"Preparing data with tokenizer: {tokenizer_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    phases_data = {}
    for phase_key, domain_config in domains.items():
        print(f"\n--- Phase {phase_key}: {domain_config.display_name} ---")
        train_dataset, val_dataset = prepare_domain_data(
            domain_config, tokenizer, train_config, seed=seed
        )
        phases_data[phase_key] = {
            "train": train_dataset,
            "val": val_dataset,
            "config": domain_config,
        }

    return phases_data, tokenizer


def create_dataloader(
    dataset: TextDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader from a TextDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,  # Keep simple for Modal
    )
