"""
avr.trainer — LEARN phase implementations.

v1 (shipped):
    SFTStrategy         — sequential SFT on LoRA, no CL.
    ReplaySFTStrategy   — SFT with replay buffer mixed in (baseline).

v2 (future):
    DPOStrategy         — DPO preference learning
    GRPOStrategy        — GRPO reinforcement learning

The framework supports the full post-training pipeline by letting users
swap the learn strategy per stage. SFT → DPO → new-domain-SFT all run
through the same LEARN → VERIFY → REPAIR loop.
"""

from __future__ import annotations
from typing import List, Optional, Dict
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .framework import LearnStrategy, TaskSpec


class TextDataset(Dataset):
    """Tokenizes (prompt, answer) pairs into a flat token stream."""

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, len(token_ids) // context_length)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        s = idx * self.context_length
        e = s + self.context_length
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}


def build_token_stream(tokenizer, pairs):
    """Flatten (prompt, answer) pairs into one token tensor."""
    all_tokens = []
    for prompt, answer in pairs:
        text = prompt + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)


class SFTStrategy(LearnStrategy):
    """Plain sequential SFT on LoRA. No CL machinery — just training.

    This is the LEARN phase. The framework handles VERIFY+REPAIR around it.
    """

    def __init__(self,
                 epochs: int = 3,
                 lr: float = 2e-4,
                 weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0,
                 batch_size: int = 8,
                 context_length: int = 512,
                 device: str = "cuda"):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device

    def train(self, model: nn.Module, task: TaskSpec, tokenizer) -> None:
        token_ids = build_token_stream(tokenizer, task.train_pairs)
        dataset = TextDataset(token_ids, self.context_length)
        print(f"    [LEARN] {len(token_ids):,} tokens, {len(dataset)} chunks")

        # Only LoRA params are trainable
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(trainable, lr=self.lr,
                                weight_decay=self.weight_decay)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        import time as _time
        t0 = _time.time()
        gs, tl = 0, 0.0
        for epoch in range(self.epochs):
            for batch in loader:
                model.train()
                out = model(
                    input_ids=batch["input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
                opt.step()
                tl += out.loss.item()
                gs += 1
                if gs % 50 == 0:
                    elapsed = _time.time() - t0
                    print(f"      step {gs} | loss={tl/gs:.4f} | {elapsed:.0f}s")
        if gs % 50 != 0:
            print(f"      step {gs} | loss={tl/gs:.4f} | done")


class ReplaySFTStrategy(LearnStrategy):
    """SFT with replay buffer mixed in. The standard CL baseline.

    For each new task, we mix in 10% (configurable) of old-task data
    into each training batch. This is the lightest replay baseline —
    if AVR can't beat this, it can't beat anything.

    The replay buffer stores raw (prompt, answer) pairs from prior tasks,
    capped at buffer_size per task. Memory grows O(N_tasks * buffer_size).
    """

    def __init__(self,
                 epochs: int = 3,
                 lr: float = 2e-4,
                 weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0,
                 batch_size: int = 8,
                 context_length: int = 512,
                 replay_ratio: float = 0.1,
                 replay_buffer_per_task: int = 200,
                 device: str = "cuda"):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.context_length = context_length
        self.replay_ratio = replay_ratio
        self.replay_buffer_per_task = replay_buffer_per_task
        self.device = device
        self.replay_buffer: List[tuple] = []  # (prompt, answer) pairs from prior tasks

    def _sample_replay(self, n: int) -> List[tuple]:
        if not self.replay_buffer:
            return []
        return random.choices(self.replay_buffer, k=min(n, len(self.replay_buffer)))

    def train(self, model: nn.Module, task: TaskSpec, tokenizer) -> None:
        # Mix current task data with replay
        current_pairs = task.train_pairs
        replay_pairs = self._sample_replay(
            int(len(current_pairs) * self.replay_ratio))
        all_pairs = current_pairs + replay_pairs
        random.shuffle(all_pairs)

        token_ids = build_token_stream(tokenizer, all_pairs)
        dataset = TextDataset(token_ids, self.context_length)
        print(f"    [LEARN/replay] {len(token_ids):,} tokens "
              f"({len(current_pairs)} current + {len(replay_pairs)} replay), "
              f"{len(dataset)} chunks")

        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]

        opt = torch.optim.AdamW(trainable, lr=self.lr,
                                weight_decay=self.weight_decay)
        loader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=True, drop_last=False)

        import time as _time
        t0 = _time.time()
        gs, tl = 0, 0.0
        for epoch in range(self.epochs):
            for batch in loader:
                model.train()
                out = model(
                    input_ids=batch["input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
                opt.step()
                tl += out.loss.item()
                gs += 1
                if gs % 50 == 0:
                    elapsed = _time.time() - t0
                    print(f"      step {gs} | loss={tl/gs:.4f} | {elapsed:.0f}s")
        if gs % 50 != 0:
            print(f"      step {gs} | loss={tl/gs:.4f} | done")

        # After training, add this task's data to the replay buffer
        if self.replay_buffer_per_task < len(task.train_pairs):
            to_add = random.sample(task.train_pairs, self.replay_buffer_per_task)
        else:
            to_add = task.train_pairs
        self.replay_buffer.extend(to_add)


# v2 stubs — interfaces reserved

class DPOStrategy(LearnStrategy):
    """v2: DPO preference learning as the LEARN phase.

    Will use TRL's DPOTrainer under the hood. Interface reserved.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("DPOStrategy is v2 research.")

    def train(self, model, task, tokenizer) -> None:
        raise NotImplementedError


class GRPOStrategy(LearnStrategy):
    """v2: GRPO reinforcement learning as the LEARN phase.

    Will use TRL's GRPOTrainer. Interface reserved.
    """
    def __init__(self, **kwargs):
        raise NotImplementedError("GRPOStrategy is v2 research.")

    def train(self, model, task, tokenizer) -> None:
        raise NotImplementedError
