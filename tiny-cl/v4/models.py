"""
V4 Models: LSTM (1.4M, Gboard-scale) + SmallGPT (30M, V1-scale).
Both trained from scratch — no pretrained base, no LoRA.
"""

import torch
import torch.nn as nn
import math
from config import LSTMConfig, SmallGPTConfig


# ──────────────────────────────────────────────
# Gboard-scale LSTM (1.4M params)
# ──────────────────────────────────────────────

class GboardLSTM(nn.Module):
    """
    1.4M parameter CIFG LSTM language model.
    Matches Gboard's production architecture:
    single layer, ~670 hidden units, embedding dim 96, 10K vocab.
    """
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
        )
        self.head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.context_length = config.context_length

    def forward(self, input_ids, labels=None):
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        logits = self.head(lstm_out)

        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def get_hidden_states(self, input_ids):
        """Return hidden states for anchor computation."""
        emb = self.embedding(input_ids)
        lstm_out, _ = self.lstm(emb)
        # Return as tuple: (embedding, lstm_output) — mimics transformer interface
        return (emb, lstm_out)


# ──────────────────────────────────────────────
# V1-scale Small GPT (30M params)
# ──────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.embed_dim // config.n_heads
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = self.attn_dropout(torch.softmax(att, dim=-1))
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class GPTBlock(nn.Module):
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallGPT(nn.Module):
    """
    30M parameter GPT trained from scratch.
    No pretrained base — full end-to-end training.
    """
    def __init__(self, config: SmallGPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.context_length = config.context_length
        self._hidden_states_cache = []

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)

        # Collect hidden states for anchors
        self._hidden_states_cache = [x.detach()]
        for block in self.blocks:
            x = block(x)
            self._hidden_states_cache.append(x.detach())

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': tuple(self._hidden_states_cache),
        })()

    def get_hidden_states(self, input_ids):
        """Return all hidden states for anchor computation."""
        with torch.no_grad():
            B, T = input_ids.shape
            pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.pos_embedding(pos)
            hiddens = [x]
            for block in self.blocks:
                x = block(x)
                hiddens.append(x)
            return tuple(hiddens)


def create_model(model_name: str, device: str = "cpu"):
    """Create a model by name."""
    if model_name == "lstm_1.4M":
        model = GboardLSTM(LSTMConfig())
    elif model_name == "gpt_30M":
        model = SmallGPT(SmallGPTConfig())
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Created {model_name}: {n_params:,} params (~{n_params * 4 / 1024 / 1024:.1f}MB)")
    return model.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
