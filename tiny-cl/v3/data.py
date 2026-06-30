"""
V3 Data Pipeline: download, tokenize, non-overlapping chunks.
"""

import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Tuple

from config import DomainConfig, TrainConfig


class TextDataset(Dataset):
    """Tokenized text dataset with non-overlapping chunks."""

    def __init__(self, token_ids: torch.Tensor, context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = len(token_ids) // context_length

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        chunk = self.token_ids[start:end]
        return {
            "input_ids": chunk,
            "labels": chunk.clone(),
        }


def prepare_domain(
    domain: DomainConfig,
    tokenizer,
    train_config: TrainConfig,
    seed: int = 42,
) -> Dict:
    print(f"  Preparing domain: {domain.display_name} ({domain.dataset_name})")

    ds_kwargs = {"path": domain.dataset_name, "split": domain.split}
    if domain.dataset_config:
        ds_kwargs["name"] = domain.dataset_config

    dataset = load_dataset(**ds_kwargs)

    texts = [t for t in dataset[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed)
    random.shuffle(texts)

    all_tokens = []
    total_tokens = 0
    max_tokens = domain.max_tokens if domain.max_tokens > 0 else float("inf")

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        if total_tokens >= max_tokens:
            break

    token_ids = torch.tensor(all_tokens[:int(max_tokens)], dtype=torch.long)
    print(f"    {len(token_ids):,} tokens ({len(token_ids) / 1e6:.1f}M)")

    n_total = len(token_ids)
    n_val = min(int(n_total * 0.1), train_config.eval_samples)
    n_train = n_total - n_val

    train_dataset = TextDataset(token_ids[:n_train], train_config.context_length)
    val_dataset = TextDataset(token_ids[n_train:n_train + n_val], train_config.context_length)

    print(f"    Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")

    return {"train": train_dataset, "val": val_dataset, "n_tokens": len(token_ids)}


def prepare_all_domains(
    domains: Dict[str, DomainConfig],
    model_hf_id: str,
    train_config: TrainConfig,
    seed: int = 42,
) -> Tuple[Dict, object]:
    print(f"\nLoading tokenizer: {model_hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    phases_data = {}
    for phase_key, domain in domains.items():
        phases_data[phase_key] = prepare_domain(domain, tokenizer, train_config, seed)

    return phases_data, tokenizer


def create_dataloader(dataset, batch_size: int = 16, shuffle: bool = True,
                      drop_last: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last, pin_memory=True)
