"""
V4 Data Pipeline: Streaming-friendly with variable increment sizes.
Tokenizes once, then serves chunks at any granularity.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple

from config import DomainConfig, TrainConfig


class TokenDataset(Dataset):
    """Simple token-level dataset. Returns (input_ids, labels) pairs."""
    def __init__(self, token_ids: torch.Tensor, context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, (len(token_ids) - 1) // context_length)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1  # +1 for label shift
        if end > len(self.token_ids):
            end = len(self.token_ids)
            start = max(0, end - self.context_length - 1)
        chunk = self.token_ids[start:end]
        # Pad if needed
        if len(chunk) < self.context_length + 1:
            pad = torch.zeros(self.context_length + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, pad])
        return {
            "input_ids": chunk[:self.context_length],
            "labels": chunk[1:self.context_length + 1],
        }


class StreamingDataset:
    """
    Serves data in increments of N examples.
    The core abstraction for the streaming experiment.
    """
    def __init__(self, token_ids: torch.Tensor, context_length: int,
                 increment_size: int = 0):
        """
        Args:
            token_ids: Flat token tensor
            context_length: Sequence length
            increment_size: Examples per increment. 0 = full phase.
        """
        self.token_ids = token_ids
        self.context_length = context_length
        self.full_dataset = TokenDataset(token_ids, context_length)
        self.increment_size = increment_size
        self.n_total = len(self.full_dataset)

        if increment_size == 0 or increment_size >= self.n_total:
            # Full phase mode
            self.n_increments = 1
            self.increment_size = self.n_total
        else:
            self.n_increments = (self.n_total + increment_size - 1) // increment_size

    def get_increment(self, inc_idx: int) -> List[Dict]:
        """Get the i-th increment as a list of {input_ids, labels} dicts."""
        start = inc_idx * self.increment_size
        end = min(start + self.increment_size, self.n_total)
        samples = []
        for idx in range(start, end):
            samples.append(self.full_dataset[idx])
        return samples

    def get_increment_tensor(self, inc_idx: int, device: str = "cpu"):
        """Get the i-th increment as batched tensors."""
        samples = self.get_increment(inc_idx)
        input_ids = torch.stack([s["input_ids"] for s in samples]).to(device)
        labels = torch.stack([s["labels"] for s in samples]).to(device)
        return {"input_ids": input_ids, "labels": labels}

    def get_full_dataset(self):
        """Get the entire dataset as a TokenDataset."""
        return self.full_dataset

    def get_probes(self, n_probes: int, device: str = "cpu"):
        """Random sample for anchor probes."""
        n_probes = min(n_probes, self.n_total)
        indices = random.sample(range(self.n_total), n_probes)
        samples = [self.full_dataset[i] for i in indices]
        input_ids = torch.stack([s["input_ids"] for s in samples]).to(device)
        return input_ids


def prepare_domain(domain: DomainConfig, tokenizer, context_length: int,
                   max_tokens: int = 0, seed: int = 42):
    """Download and tokenize a single domain."""
    print(f"  Preparing: {domain.display_name} ({domain.dataset_name})")

    ds_kwargs = {"path": domain.dataset_name, "split": domain.split}
    dataset = load_dataset(**ds_kwargs)

    texts = [t for t in dataset[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed)
    random.shuffle(texts)

    all_tokens = []
    total = 0
    limit = max_tokens if max_tokens > 0 else float("inf")
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        total += len(tokens)
        if total >= limit:
            break

    token_ids = torch.tensor(all_tokens[:int(limit)], dtype=torch.long)
    print(f"    {len(token_ids):,} tokens ({len(token_ids)/1e6:.1f}M)")

    # 90/10 split
    n_val = min(int(len(token_ids) * 0.1), 100_000)
    n_train = len(token_ids) - n_val

    return {
        "train_tokens": token_ids[:n_train],
        "val_tokens": token_ids[n_train:n_train + n_val],
    }


def prepare_all_domains(domains, vocab_size=10000, context_length=256,
                        max_tokens=500_000, seed=42):
    """Build tokenizer and prepare all domains."""
    # Train a simple BPE tokenizer on combined text
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    print("\nTraining tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"])

    # Collect text from all domains for tokenizer training
    all_texts = []
    for domain in domains.values():
        ds = load_dataset(path=domain.dataset_name, split=domain.split)
        texts = [t for t in ds[domain.text_field] if t and len(t.strip()) > 10][:5000]
        all_texts.extend(texts)

    tokenizer.train_from_iterator(all_texts, trainer)

    # Wrap tokenizer for encode/decode
    class TokenizerWrapper:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.pad_token_id = 1
        def encode(self, text, add_special_tokens=False):
            return self.tokenizer.encode(text).ids
        def decode(self, ids):
            return self.tokenizer.decode(ids)
        def batch_decode(self, batch):
            return [self.decode(ids) for ids in batch]

    tok_wrapper = TokenizerWrapper(tokenizer)

    phases_data = {}
    for phase_key, domain in domains.items():
        data = prepare_domain(domain, tok_wrapper, context_length, max_tokens, seed)
        phases_data[phase_key] = data

    return phases_data, tok_wrapper
