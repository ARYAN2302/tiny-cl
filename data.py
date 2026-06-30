"""
Data loading, tokenization, and phase splitting.
Downloads from HuggingFace, tokenizes with GPT-2 tokenizer,
and creates DataLoaders for each phase.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
from config import PhaseConfig, TrainConfig
from tqdm import tqdm


class TextDataset(Dataset):
    """Simple dataset for tokenized text sequences."""
    
    def __init__(self, token_ids: torch.Tensor, context_length: int):
        """
        Args:
            token_ids: 1D tensor of all token IDs
            context_length: length of each sequence
        """
        self.token_ids = token_ids
        self.context_length = context_length
    
    def __len__(self):
        return max(0, (len(self.token_ids) - 1) // self.context_length)
    
    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1  # +1 for labels shift
        chunk = self.token_ids[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return {"input_ids": x, "labels": y}


def get_tokenizer() -> GPT2Tokenizer:
    """Load GPT-2 tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def count_words(text: str) -> int:
    """Rough word count."""
    return len(text.split())


def load_phase_data(
    phase_config: PhaseConfig,
    tokenizer: GPT2Tokenizer,
    train_config: TrainConfig,
    cache_dir: str = None,
) -> tuple:
    """
    Download and tokenize data for a single phase.
    
    Returns:
        (train_dataset, val_dataset, word_count)
    """
    print(f"\n{'='*60}")
    print(f"Loading Phase: {phase_config.name}")
    print(f"Dataset: {phase_config.dataset_name} ({phase_config.dataset_config})")
    print(f"{'='*60}")
    
    # Load dataset
    kwargs = {"path": phase_config.dataset_name}
    if phase_config.dataset_config:
        kwargs["name"] = phase_config.dataset_config
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    
    # Try train split
    try:
        dataset = load_dataset(**kwargs, split="train")
    except Exception:
        dataset = load_dataset(**kwargs, split="train[:100%]")
    
    # Extract text and limit by word count
    texts = []
    total_words = 0
    max_words = phase_config.max_words
    
    if train_config.debug:
        max_words = min(max_words, 50_000)  # Tiny subset for debugging
    
    for example in tqdm(dataset, desc=f"Reading {phase_config.name}"):
        text = example.get(phase_config.text_field, "")
        if not text or len(text.strip()) < 10:  # Skip empty/very short lines
            continue
        
        texts.append(text)
        total_words += count_words(text)
        
        if max_words > 0 and total_words >= max_words:
            break
    
    print(f"  Loaded {len(texts)} texts, ~{total_words:,} words")
    
    # Split 90/10 train/val
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Tokenize
    print(f"  Tokenizing training data...")
    train_ids = tokenize_texts(train_texts, tokenizer)
    print(f"  Tokenizing validation data...")
    val_ids = tokenize_texts(val_texts, tokenizer)
    
    print(f"  Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")
    
    # Create datasets
    train_dataset = TextDataset(train_ids, train_config.context_length)
    val_dataset = TextDataset(val_ids, train_config.context_length)
    
    print(f"  Train sequences: {len(train_dataset):,} | Val sequences: {len(val_dataset):,}")
    
    return train_dataset, val_dataset, total_words


def tokenize_texts(texts: list, tokenizer: GPT2Tokenizer) -> torch.Tensor:
    """Tokenize a list of texts into a single 1D tensor of token IDs."""
    all_ids = []
    
    for text in tqdm(texts, desc="Tokenizing", leave=False):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(tokenizer.eos_token_id)
    
    return torch.tensor(all_ids, dtype=torch.long)


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
        num_workers=0,  # Simple for research code
        pin_memory=True,
        drop_last=drop_last,
    )


def load_all_phases(
    phase_configs: dict,
    tokenizer: GPT2Tokenizer,
    train_config: TrainConfig,
    cache_dir: str = None,
) -> dict:
    """
    Load data for all phases.
    
    Returns:
        {
            "A": {"train": DataLoader, "val": DataLoader, "train_dataset": Dataset, "val_dataset": Dataset, "word_count": int},
            "B": {...},
            "C": {...},
        }
    """
    phases = {}
    
    for phase_key, phase_config in phase_configs.items():
        train_dataset, val_dataset, word_count = load_phase_data(
            phase_config, tokenizer, train_config, cache_dir
        )
        
        phases[phase_key] = {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "train_loader": create_dataloader(train_dataset, train_config.batch_size, shuffle=True, drop_last=True),
            "val_loader": create_dataloader(val_dataset, train_config.batch_size, shuffle=False, drop_last=False),
            "word_count": word_count,
        }
    
    return phases
