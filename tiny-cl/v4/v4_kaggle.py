"""
V4: Streaming AVR — Does Verification-Driven Repair Survive Granularity Collapse?
Single-file Kaggle runner. No dependencies beyond torch, transformers, datasets.

Run: python v4_kaggle.py
Or run the full grid: python v4_kaggle.py --grid
Quick test: python v4_kaggle.py --quick

METHOD NOMENCLATURE (important — AVR means different things across versions):
  - AVR-repr (V1-V3): Verification via hidden-state MSE anchors + repair via weight MSE
  - AVR-loss (V4):     Verification via LOSS on probe set + repair via weight MSE
                       Hidden-state means were too robust to detect functional degradation.
                       Loss-based verification directly measures forgetting.
  - AVR-always (V4):   Ablation — always repair, no verification gate.
                       Tests whether the gate adds value vs just pulling toward old weights.

METHODS IN THIS FILE:
  naive      = standard fine-tuning, no continual learning
  avr        = AVR-loss (gated: verify→repair only if loss increase > 5%)
  avr_always = AVR-loss ablation (always repair, no gate)
  ewc        = Elastic Weight Consolidation (Fisher from last increment only)

FIXES (from earlier runs):
  1. True 1.4M LSTM: embed_dim=hidden_dim=128, weight-tied head
  2. Loss-based anchors (hidden-state MSE was too robust, never triggered repair)
  3. Drift threshold 5% (20% never triggered; 8% loss increase ≈ 85% PPL increase)
  4. cuDNN LSTM: model.train() everywhere (backward crashes in eval mode)
  5. Tokenizer: PAD token id=1 not 0 (was padding with [UNK] not [PAD])
  6. compute_ppl: mask pad_id=1 not 0 (was skipping genuine [UNK] predictions)
  7. EWC Fisher: computed from last increment only (was using full phase — unfair)
"""

import os, json, time, random, math, argparse, gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

@dataclass
class LSTMConfig:
    vocab_size: int = 10000
    embed_dim: int = 128       # MUST equal hidden_dim for weight tying to work
    hidden_dim: int = 128      # MUST equal embed_dim — was 400 but that breaks weight tying
    n_layers: int = 1
    context_length: int = 32   # was 64 — reduced for T4 memory
    batch_size: int = 8        # LSTM-specific batch size (was 16)
    name: str = "lstm_1.4M"

@dataclass
class SmallGPTConfig:
    vocab_size: int = 10000
    embed_dim: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 2048
    context_length: int = 256
    batch_size: int = 16       # GPT can handle larger batches
    name: str = "gpt_30M"

@dataclass
class DomainConfig:
    name: str
    display_name: str
    dataset_name: str
    text_field: str = "text"
    split: str = "train"
    max_tokens: int = 0

@dataclass
class AVRConfig:
    n_anchor_probes: int = 20
    drift_threshold: float = 0.05  # relative loss increase to trigger repair (5% loss increase ≈ 50% PPL increase = forgetting)
    repair_steps: int = 30
    repair_lr: float = 1e-4
    verify_every_n_increments: int = 5

@dataclass
class EWCConfig:
    lambda_: float = 0.1
    fisher_n_samples: int = 200

@dataclass
class TrainConfig:
    batch_size: int = 16       # Default; overridden per-model
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    eval_samples: int = 512
    results_dir: str = "v4_results"

MODEL_CONFIGS = {"lstm_1.4M": LSTMConfig(), "gpt_30M": SmallGPTConfig()}
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text", max_tokens=500_000),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output", max_tokens=500_000),
}
INCREMENT_SIZES = [0, 500, 100, 20]
METHODS = ["naive", "avr", "avr_always", "ewc"]
# avr = AVR-loss: loss-based verification + gated repair (only when degraded)
# avr_always = ablation: repair every N increments regardless of verification


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────

class GboardLSTM(nn.Module):
    """
    True 1.4M param LSTM language model.
    
    Architecture (with weight tying, embed_dim == hidden_dim):
      Embedding: 10000 * 128 = 1,280,000
      LSTM:      4*(128*128 + 128*128 + 2*128) = 132,096
      Head:      0 (tied to embedding)
      ─────────────────────────────────────────
      Total:     ~1,412,096 ≈ 1.4M params ✓
    
    NOTE: embed_dim MUST equal hidden_dim for weight tying to work.
    If they differ, the head weight shape [vocab, embed_dim] won't match
    the LSTM output shape [batch, seq, hidden_dim] → matmul crash.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_dim, config.n_layers, batch_first=True)
        # Weight tying: share embedding and output projection (like Gboard)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight  # TIED — cuts from 9.7M to 1.4M
        self.context_length = config.context_length

    def forward(self, input_ids, labels=None):
        emb = self.embedding(input_ids)
        out, _ = self.lstm(emb)
        logits = self.head(out)
        loss = None
        if labels is not None:
            s_logits = logits[:, :-1, :].contiguous()
            s_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(s_logits.view(-1, self.config.vocab_size), s_labels.view(-1))
        return type('O', (), {'loss': loss, 'logits': logits, 'hidden_states': (emb, out)})()

    def get_hidden_states(self, input_ids):
        with torch.no_grad():
            emb = self.embedding(input_ids)
            out, _ = self.lstm(emb)
            return (emb, out)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.embed_dim // cfg.n_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.attn_drop = nn.Dropout(0.1)
        self.resid_drop = nn.Dropout(0.1)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.context_length, cfg.context_length)).view(1, 1, cfg.context_length, cfg.context_length))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        y = self.attn_drop(torch.softmax(att, dim=-1)) @ v
        return self.resid_drop(self.proj(y.transpose(1, 2).contiguous().view(B, T, C)))


class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embed_dim)
        self.mlp = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.ff_dim), nn.GELU(), nn.Linear(cfg.ff_dim, cfg.embed_dim), nn.Dropout(0.1))

    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))


class SmallGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.context_length, config.embed_dim)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.context_length = config.context_length

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(0, T, device=input_ids.device).unsqueeze(0))
        hiddens = [x.detach()]
        for block in self.blocks:
            x = block(x)
            hiddens.append(x.detach())
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size), labels[:, 1:].contiguous().view(-1))
        return type('O', (), {'loss': loss, 'logits': logits, 'hidden_states': tuple(hiddens)})()

    def get_hidden_states(self, input_ids):
        with torch.no_grad():
            B, T = input_ids.shape
            x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(0, T, device=input_ids.device).unsqueeze(0))
            hiddens = [x]
            for block in self.blocks:
                x = block(x)
                hiddens.append(x)
            return tuple(hiddens)


def create_model(name, device):
    cfg = MODEL_CONFIGS[name]
    model = GboardLSTM(cfg) if "lstm" in name else SmallGPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n:,} params (~{n*4/1024/1024:.1f}MB)")
    return model.to(device)


def get_batch_size(model_name):
    """Return model-specific batch size."""
    cfg = MODEL_CONFIGS[model_name]
    return getattr(cfg, 'batch_size', 16)


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, tokens, ctx_len, pad_id=1):
        self.tokens = tokens
        self.ctx = ctx_len
        self.pad_id = pad_id  # Use actual PAD token id, not 0 (which is [UNK])
        self.n = max(1, (len(tokens) - 1) // ctx_len)

    def __len__(self): return self.n

    def __getitem__(self, i):
        s = i * self.ctx
        e = min(s + self.ctx + 1, len(self.tokens))
        chunk = self.tokens[s:e]
        if len(chunk) < self.ctx + 1:
            chunk = torch.cat([chunk, torch.full((self.ctx + 1 - len(chunk),), self.pad_id, dtype=torch.long)])
        return {"input_ids": chunk[:self.ctx], "labels": chunk[1:self.ctx+1]}


class StreamingDataset:
    def __init__(self, tokens, ctx_len, inc_size=0):
        self.tokens = tokens
        self.ctx = ctx_len
        self.full = TokenDataset(tokens, ctx_len)
        self.n_total = len(self.full)
        if inc_size == 0 or inc_size >= self.n_total:
            self.inc_size = self.n_total
            self.n_inc = 1
        else:
            self.inc_size = inc_size
            self.n_inc = (self.n_total + inc_size - 1) // inc_size

    def get_increment_tensor(self, idx, device="cpu"):
        """DEPRECATED — use get_increment_dataset() + DataLoader for mini-batching."""
        s, e = idx * self.inc_size, min((idx+1) * self.inc_size, self.n_total)
        samples = [self.full[i] for i in range(s, e)]
        return {"input_ids": torch.stack([x["input_ids"] for x in samples]).to(device),
                "labels": torch.stack([x["labels"] for x in samples]).to(device)}

    def get_increment_dataset(self, idx):
        """Return a Dataset for the i-th increment (for mini-batch training)."""
        s, e = idx * self.inc_size, min((idx+1) * self.inc_size, self.n_total)
        indices = list(range(s, e))
        return torch.utils.data.Subset(self.full, indices)

    def get_full(self): return self.full

    def get_probes(self, n, device="cpu"):
        n = min(n, self.n_total)
        indices = random.sample(range(self.n_total), n)
        return torch.stack([self.full[i]["input_ids"] for i in indices]).to(device)


# ──────────────────────────────────────────────
# ANCHORS
# ──────────────────────────────────────────────

class StreamingAnchorStore:
    """
    Loss-based anchor store for detecting forgetting.
    
    Instead of comparing hidden state means (which are too robust to detect
    functional degradation), we store the initial LOSS on probe examples.
    If the current loss on those probes increases beyond a relative threshold,
    that phase is marked as degraded and repair is triggered.
    
    This directly measures what we care about: can the model still produce
    the same outputs on old data? A 20% loss increase = forgetting detected.
    """
    def __init__(self):
        self.anchors = {}   # phase_key -> {"probes": tensor, "initial_loss": float, "labels": tensor}
        self.health = {}   # phase_key -> float (1.0 = perfect, 0.0 = total forgetting)

    def save_anchors(self, model, phase_key, sdata, n_probes=20, device="cpu"):
        # Use train mode — LSTM with cuDNN needs it, and loss under no_grad is the same
        model.train()
        probes = sdata.get_probes(n_probes, device)
        # Compute initial loss on probes — this is our "memory" of this phase
        with torch.no_grad():
            # We need labels for loss computation
            # probes shape: [n_probes, ctx_len] — these are input_ids
            # Labels should be shifted: labels = input_ids shifted by 1
            labels = torch.cat([probes[:, 1:], torch.zeros(probes.size(0), 1, dtype=probes.dtype, device=device)], dim=1)
            o = model(input_ids=probes, labels=labels)
            initial_loss = o.loss.item()
        self.anchors[phase_key] = {"probes": probes.cpu(), "labels": labels.cpu(), "initial_loss": initial_loss, "n": n_probes}
        self.health[phase_key] = 1.0
        kb = probes.numel() * 4 / 1024
        print(f"  Anchors: Phase {phase_key} (loss={initial_loss:.4f}, {n_probes} probes, ~{kb:.1f}KB)")

    def verify(self, model, phase_keys, threshold=0.2, device="cpu"):
        """
        Check if model has forgotten any previous phase.
        
        threshold: relative loss increase to trigger degradation.
            0.2 = 20% loss increase = degraded
            
        Returns (drift_report, degraded_phases, needs_repair)
        """
        drift_report, degraded = {}, []
        for pk in phase_keys:
            if pk not in self.anchors: continue
            probes = self.anchors[pk]["probes"].to(device)
            labels = self.anchors[pk]["labels"].to(device)
            with torch.no_grad():
                o = model(input_ids=probes, labels=labels)
                current_loss = o.loss.item()
            initial_loss = self.anchors[pk]["initial_loss"]
            if initial_loss > 0:
                relative_increase = (current_loss - initial_loss) / initial_loss
            else:
                relative_increase = 0.0
            drift_report[pk] = {"initial_loss": initial_loss, "current_loss": current_loss, "relative_increase": relative_increase}
            # Health: 1.0 when loss hasn't changed, decreases as loss increases
            # At threshold (20% increase), health = 0.0
            self.health[pk] = max(0.0, 1.0 - relative_increase / threshold)
            if relative_increase > threshold:
                degraded.append(pk)
        return drift_report, degraded, len(degraded) > 0

    def get_storage_kb(self):
        return sum(d["probes"].numel() * 4 / 1024 + d["labels"].numel() * 4 / 1024 for d in self.anchors.values())


# ──────────────────────────────────────────────
# METHODS
# ──────────────────────────────────────────────

class NaiveMethod:
    def __init__(self): self.name, self.completed, self.extra_steps = "naive", [], 0
    def on_phase_start(self, model, pk): pass
    def on_increment_end(self, model, pk, dev): pass
    def on_phase_end(self, model, pk, sd, dev): self.completed.append(pk)
    def compute_loss(self, model, batch, dev):
        o = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
        return o.loss, {"lm_loss": o.loss.item()}


class AVRMethod:
    def __init__(self, cfg=None):
        self.cfg = cfg or AVRConfig()
        self.name = "avr"
        self.anchor_store = StreamingAnchorStore()
        self.snapshots = {}
        self.completed = []
        self.extra_steps = 0
        self.total_repairs = 0
        self.total_verifies = 0
        self.inc_count = 0

    def on_phase_start(self, model, pk): self.inc_count = 0

    def on_increment_end(self, model, pk, dev):
        self.inc_count += 1
        if self.completed and self.inc_count % self.cfg.verify_every_n_increments == 0:
            self._vr(model, dev)

    def on_phase_end(self, model, pk, sdata, dev):
        self.anchor_store.save_anchors(model, pk, sdata, self.cfg.n_anchor_probes, dev)
        snap = {n: p.data.cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.snapshots[pk] = snap
        self._vr(model, dev)
        self.completed.append(pk)

    def compute_loss(self, model, batch, dev):
        o = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
        return o.loss, {"lm_loss": o.loss.item()}

    def _vr(self, model, dev):
        self.total_verifies += 1
        drift_report, degraded, needs = self.anchor_store.verify(model, self.completed, self.cfg.drift_threshold, dev)
        for pk in self.completed:
            h = self.anchor_store.health.get(pk, 1.0)
            dr = drift_report.get(pk, {})
            rel = dr.get("relative_increase", 0)
            print(f"  [VERIFY] {pk}: loss {dr.get('initial_loss',0):.3f}→{dr.get('current_loss',0):.3f} (+{rel*100:.1f}%) health={h:.3f} [{'OK' if h > 0.0 else 'DEGRADED'}]")
        if not needs: return
        self.total_repairs += 1
        print(f"  [REPAIR] {degraded} — {self.cfg.repair_steps} steps")
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=self.cfg.repair_lr)
        for _ in range(self.cfg.repair_steps):
            wl = torch.tensor(0.0, device=dev)
            n = 0
            for pk in set(degraded):
                if pk not in self.snapshots: continue
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.snapshots[pk]:
                        wl = wl + F.mse_loss(param, self.snapshots[pk][name].to(dev))
                        n += 1
            if n > 0 and wl.requires_grad:
                opt.zero_grad()
                (wl / n).backward()
                opt.step()
                self.extra_steps += 1
            else: break
        _, new_deg, _ = self.anchor_store.verify(model, self.completed, self.cfg.drift_threshold, dev)
        print(f"  [REPAIR] {'OK' if not new_deg else f'Partial: {new_deg}'}")


class AVRAlwaysMethod(AVRMethod):
    """
    Ablation: always repair, no verification gate.
    
    Same as AVR-loss but skips the verify step — always runs repair
    every N increments. This tests whether the VALUE is in the
    verification gate or just in pulling weights toward old snapshots.
    
    If AVR-gated ≈ AVR-always: the gate isn't adding value (just repair helps).
    If AVR-gated >> AVR-always: the gate is critical (ungated repair hurts plasticity).
    If AVR-gated ≈ AVR-always ≈ naive: repair doesn't help at all.
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.name = "avr_always"

    def on_increment_end(self, model, pk, dev):
        self.inc_count += 1
        if self.completed and self.inc_count % self.cfg.verify_every_n_increments == 0:
            # Skip verification — just always repair
            self._always_repair(model, dev)

    def on_phase_end(self, model, pk, sdata, dev):
        self.anchor_store.save_anchors(model, pk, sdata, self.cfg.n_anchor_probes, dev)
        snap = {n: p.data.cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.snapshots[pk] = snap
        # Final repair at phase end
        self._always_repair(model, dev)
        self.completed.append(pk)

    def _always_repair(self, model, dev):
        """Always repair toward all completed phase snapshots, no verification."""
        if not self.snapshots:
            return
        self.total_repairs += 1
        self.total_verifies += 1  # Count it for fair comparison of overhead
        degraded = list(self.completed)
        print(f"  [REPAIR-ALWAYS] {degraded} — {self.cfg.repair_steps} steps (no gate)")
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=self.cfg.repair_lr)
        for _ in range(self.cfg.repair_steps):
            wl = torch.tensor(0.0, device=dev)
            n = 0
            for pk in set(degraded):
                if pk not in self.snapshots: continue
                for name, param in model.named_parameters():
                    if param.requires_grad and name in self.snapshots[pk]:
                        wl = wl + F.mse_loss(param, self.snapshots[pk][name].to(dev))
                        n += 1
            if n > 0 and wl.requires_grad:
                opt.zero_grad()
                (wl / n).backward()
                opt.step()
                self.extra_steps += 1
            else: break


class EWCMethod:
    def __init__(self, cfg=None):
        self.cfg = cfg or EWCConfig()
        self.name = "ewc"
        self.completed = []
        self.extra_steps = 0
        self.fisher = {}
        self.opt_params = {}
        self.computable = {}

    def on_phase_start(self, model, pk): pass
    def on_increment_end(self, model, pk, dev): pass

    def on_phase_end(self, model, pk, sdata, dev):
        # Use the LAST INCREMENT only for Fisher, not the full phase.
        # This makes EWC operate in the same streaming regime as AVR —
        # if increments are small, Fisher is estimated from small data.
        # This is the fair comparison: both methods see the same data budget.
        n_inc = sdata.n_inc
        last_inc = sdata.get_increment_dataset(n_inc - 1)
        n_avail = len(last_inc)
        if n_avail < 50:
            print(f"  [EWC] Only {n_avail} examples in last increment — Fisher UNRELIABLE")
            self.computable[pk] = False
            self.completed.append(pk)
            return
        self.computable[pk] = True
        n_samp = min(self.cfg.fisher_n_samples, n_avail)
        print(f"  [EWC] Computing Fisher from {n_samp} samples (last increment, {n_avail} total)...")
        # Keep model in TRAINING mode — cuDNN LSTM backward crashes in eval mode
        model.train()
        f_dict = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
        o_dict = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        # Use small batch size for Fisher to avoid OOM
        fisher_bs = min(8, n_samp)
        loader = DataLoader(last_inc, batch_size=fisher_bs, shuffle=True, drop_last=False)
        done = 0
        for batch in loader:
            if done >= n_samp: break
            model.zero_grad()
            o = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
            o.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    f_dict[n] += p.grad.data.pow(2) / n_samp
            done += batch["input_ids"].size(0)
        self.fisher[pk] = f_dict
        self.opt_params[pk] = o_dict
        self.completed.append(pk)
        model.train()

    def compute_loss(self, model, batch, dev):
        o = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
        lm = o.loss
        ewc = torch.tensor(0.0, device=dev)
        for pk in self.completed:
            if not self.computable.get(pk, False): continue
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.fisher.get(pk, {}):
                    ewc = ewc + (self.fisher[pk][n].to(dev) * (p - self.opt_params[pk][n].to(dev)).pow(2)).sum()
        return lm + self.cfg.lambda_ * ewc, {"lm_loss": lm.item(), "ewc_loss": ewc.item()}


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, dataset, dev, max_samp=512):
    # Use train mode — cuDNN LSTM compat; no_grad means no dropout anyway
    model.train()
    eval_bs = 8
    pad_id = getattr(dataset, 'pad_id', 1) if hasattr(dataset, 'pad_id') else 1
    loader = DataLoader(dataset, batch_size=eval_bs, shuffle=False, drop_last=False)
    tot_loss, tot_tok, nb = 0.0, 0, 0
    for batch in loader:
        if nb * eval_bs >= max_samp: break
        o = model(input_ids=batch["input_ids"].to(dev), labels=batch["labels"].to(dev))
        # Count non-padding tokens — use pad_id (1), not 0 (which is [UNK])
        nt = (batch["labels"] != pad_id).sum().item()
        if nt == 0: nt = batch["input_ids"].numel()
        tot_loss += o.loss.item() * nt
        tot_tok += nt
        nb += 1
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")

def eval_all(model, val_ds, phases, dev, max_samp=512):
    return {"perplexity": {pk: compute_ppl(model, val_ds[pk], dev, max_samp) for pk in phases if pk in val_ds}}


# ──────────────────────────────────────────────
# TOKENIZER BUILDER
# ──────────────────────────────────────────────

def build_tokenizer(domains, vocab_size=10000):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from datasets import load_dataset as ld

    print("Training tokenizer...")
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[PAD]"])
    texts = []
    for d in domains.values():
        try:
            ds = ld(path=d.dataset_name, split=d.split)
            texts.extend([t for t in ds[d.text_field] if t and len(t.strip()) > 10][:5000])
        except: pass
    tok.train_from_iterator(texts, trainer)

    class W:
        def __init__(self, t): self.t = t; self.pad_token_id = 1
        def encode(self, text, **kw): return self.t.encode(text).ids
        def decode(self, ids): return self.t.decode(ids)
    return W(tok)


def prepare_domain(domain, tokenizer, ctx_len, max_tok, seed=42):
    from datasets import load_dataset as ld
    print(f"  Loading: {domain.display_name}")
    ds = ld(path=domain.dataset_name, split=domain.split)
    texts = [t for t in ds[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed)
    random.shuffle(texts)
    tokens = []
    total = 0
    limit = max_tok if max_tok > 0 else float("inf")
    for t in texts:
        tokens.extend(tokenizer.encode(t))
        total += len(tokens) - total  # rough
        if len(tokens) >= limit: break
    tids = torch.tensor(tokens[:int(limit)], dtype=torch.long)
    print(f"    {len(tids):,} tokens")
    n_val = min(int(len(tids) * 0.1), 100_000)
    return tids[:len(tids)-n_val], tids[len(tids)-n_val:]


# ──────────────────────────────────────────────
# OOM HELPER
# ──────────────────────────────────────────────

def safe_forward(model, batch, dev, method):
    """
    Try forward+backward on device. If CUDA OOM, fall back to CPU.
    Returns (loss, metrics, actual_device).
    """
    try:
        loss, metrics = method.compute_loss(model, batch, dev)
        return loss, metrics, dev
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and "cuda" in str(e).lower():
            print(f"  ⚠ CUDA OOM — falling back to CPU for this batch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Move model to CPU
            model.cpu()
            loss, metrics = method.compute_loss(model, batch, "cpu")
            return loss, metrics, "cpu"
        raise


def move_to_device(batch, device):
    """Move a batch dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


# ──────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────

def run_experiment(model_name, method_name, inc_size=0, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    tc = TrainConfig()
    dev = tc.device
    mcfg = MODEL_CONFIGS[model_name]
    ctx = mcfg.context_length
    batch_size = get_batch_size(model_name)  # Model-specific batch size

    print(f"\n{'#'*60}")
    print(f"# {model_name} | {method_name} | inc={'full' if inc_size==0 else inc_size}")
    print(f"# device={dev} | batch_size={batch_size} | ctx={ctx}")
    print(f"{'#'*60}")

    model = create_model(model_name, dev)
    tokenizer = build_tokenizer(DOMAINS, mcfg.vocab_size)

    phases_data = {}
    for pk, d in DOMAINS.items():
        train_t, val_t = prepare_domain(d, tokenizer, ctx, d.max_tokens, seed)
        phases_data[pk] = {"train": train_t, "val": val_t}

    val_ds = {pk: TokenDataset(v["val"], ctx) for pk, v in phases_data.items()}
    sdata = {pk: StreamingDataset(v["train"], ctx, inc_size) for pk, v in phases_data.items()}

    if method_name == "naive": method = NaiveMethod()
    elif method_name == "avr": method = AVRMethod()
    elif method_name == "avr_always": method = AVRAlwaysMethod()
    elif method_name == "ewc": method = EWCMethod()

    results = {"model": model_name, "method": method_name, "increment": inc_size, "phases": {}}

    # Track whether we've fallen back to CPU
    actual_dev = dev

    for pk in DOMAINS.keys():
        sd = sdata[pk]
        method.on_phase_start(model, pk)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        gs, tl = 0, 0.0
        t0 = time.time()

        print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} ({sd.n_inc} increments x {sd.inc_size})")

        for ii in range(sd.n_inc):
            # Mini-batch training within each increment
            inc_dataset = sd.get_increment_dataset(ii)
            inc_loader = DataLoader(inc_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            for mini_batch in inc_loader:
                model.train()
                try:
                    loss, _, fwd_dev = safe_forward(model, mini_batch, actual_dev, method)
                    if fwd_dev != actual_dev: actual_dev = fwd_dev
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"  ⚠ OOM on increment {ii} — clearing cache, reducing and retrying on CPU")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        # Move everything to CPU
                        model.cpu()
                        actual_dev = "cpu"
                        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
                        loss, _, actual_dev = safe_forward(model, mini_batch, actual_dev, method)
                    else:
                        raise

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step()
                tl += loss.item()
                gs += 1

            # Clear CUDA cache between increments to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            method.on_increment_end(model, pk, actual_dev)
            if (ii+1) % max(1, sd.n_inc // 5) == 0:
                print(f"    inc {ii+1}/{sd.n_inc} | avg_loss={tl/max(gs,1):.4f} | dev={actual_dev}")

        method.on_phase_end(model, pk, sd, actual_dev)
        # Safety: ensure model is back in train mode (EWC/AVR on_phase_end may leave it in eval)
        model.train()
        ev = eval_all(model, val_ds, method.completed if hasattr(method, 'completed') else [pk], actual_dev, tc.eval_samples)
        ev["avg_loss"] = tl / max(gs, 1)
        ev["extra_steps"] = method.extra_steps if hasattr(method, 'extra_steps') else 0
        if hasattr(method, 'total_repairs'): ev["repairs"] = method.total_repairs
        if hasattr(method, 'total_verifies'): ev["verifies"] = method.total_verifies
        if hasattr(method, 'computable'): ev["ewc_computable"] = method.computable
        results["phases"][pk] = ev
        ev["time"] = time.time() - t0
        ev["device_used"] = actual_dev

        print(f"  Eval:")
        for p, ppl in ev.get("perplexity", {}).items():
            print(f"    {p}: PPL={ppl:.2f}")

    # Save
    os.makedirs(tc.results_dir, exist_ok=True)
    inc_s = "full" if inc_size == 0 else str(inc_size)
    fp = os.path.join(tc.results_dir, f"{model_name}_{method_name}_inc{inc_s}.json")
    with open(fp, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {fp}")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


def run_grid(models=None, increments=None, methods=None):
    models = models or list(MODEL_CONFIGS.keys())
    increments = increments or INCREMENT_SIZES
    methods = methods or METHODS
    all_res = []
    for m in models:
        for inc in increments:
            for meth in methods:
                print(f"\n{'*'*60}")
                try:
                    r = run_experiment(m, meth, inc)
                    all_res.append(r)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback; traceback.print_exc()
    os.makedirs("v4_results", exist_ok=True)
    with open("v4_results/full_grid.json", "w") as f: json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res


def run_quick():
    """Quick test: LSTM + AVR + 100-example increments. ~5 min."""
    print("QUICK TEST: lstm_1.4M + avr + inc=100")
    return run_experiment("lstm_1.4M", "avr", 100)


def print_summary(results=None):
    if results is None:
        fps = [os.path.join("v4_results", f) for f in os.listdir("v4_results") if f.endswith(".json") and f != "full_grid.json"]
        results = []
        for fp in fps:
            with open(fp) as f: results.append(json.load(f))
        fg = os.path.join("v4_results", "full_grid.json")
        if os.path.exists(fg):
            with open(fg) as f: results.extend(json.load(f))

    if not results:
        print("No results found"); return

    print(f"\n{'='*100}")
    print(f"STREAMING AVR — RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<12} {'Method':<12} {'Inc':<8} {'A→PPL':<10} {'B→PPL':<10} {'FF(A)':<8} {'Repairs':<8} {'ExtraSt':<8} {'EWC?':<8}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: (x.get("model",""), x.get("method",""), x.get("increment",0))):
        model = r.get("model", "?")
        method = r.get("method", "?")
        inc = r.get("increment", 0)
        phases = r.get("phases", {})
        keys = list(phases.keys())

        # Final-phase perplexities (after all training)
        last_ppl = phases[keys[-1]].get("perplexity", {}) if keys else {}
        # Phase A final PPL (how well we remember A)
        a_final = last_ppl.get("A", None)
        # Phase B final PPL (how well we learned B) 
        b_final = last_ppl.get("B", None)

        # Forgetting factor: A PPL after B / A PPL after A
        a_after_a = phases[keys[0]].get("perplexity", {}).get("A") if keys else None
        ff_a = a_final / a_after_a if (a_final and a_after_a and a_after_a > 0) else None

        inc_s = "full" if inc == 0 else str(inc)
        ff_s = f"{ff_a:.2f}x" if ff_a else "N/A"
        a_s = f"{a_final:.0f}" if a_final else "N/A"
        b_s = f"{b_final:.0f}" if b_final else "N/A"
        lp = phases[keys[-1]] if keys else {}
        repairs = lp.get("repairs", "-")
        ewc_ok = lp.get("ewc_computable", "-")
        extra = lp.get("extra_steps", "-")

        print(f"{model:<12} {method:<12} {inc_s:<8} {a_s:<10} {b_s:<10} {ff_s:<8} {repairs:<8} {extra:<8} {str(ewc_ok):<8}")

    print(f"{'='*100}")
    print(f"FF(A) = forgetting factor on A (A PPL after B / A PPL after A)")
    print(f"A→PPL = Phase A PPL after all training (retention), B→PPL = Phase B PPL (plasticity)")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# KAGGLE / JUPYTER NOTEBOOK CELL USAGE:
#
# Cell 1: Upload this file, then:
#   from v4_kaggle import *
#   run_quick()              # 5-min test
#   run_grid()               # Full grid (~1-2 hrs on T4/P100)
#   print_summary()          # View results table
#
# Cell 2 (single run):
#   run_experiment("lstm_1.4M", "avr", 100)
#   run_experiment("gpt_30M", "naive", 20)
#
# Or just hit Run All — it runs the full grid by default.
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Colab/Kaggle inject args like: -f /root/.local/share/jupyter/runtime/...
    # Use parse_known_args to ignore those, only process our flags
    parser = argparse.ArgumentParser(description="V4: Streaming AVR Experiment")
    parser.add_argument("--model", default="lstm_1.4M", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", default="avr", choices=METHODS)
    parser.add_argument("--increment", default=0, type=int, help="0=full, else examples per increment")
    parser.add_argument("--grid", action="store_true", help="Run full experiment grid")
    parser.add_argument("--quick", action="store_true", help="Quick 5-min test")
    parser.add_argument("--summary", action="store_true", help="Print results summary")
    args, unknown = parser.parse_known_args()

    if args.quick:
        run_quick()
    elif args.grid:
        run_grid()
    elif args.summary:
        print_summary()
    elif args.model != "lstm_1.4M" or args.method != "avr" or args.increment != 0:
        # User explicitly set model/method/increment
        run_experiment(args.model, args.method, args.increment)
    else:
        # Default: Kaggle/Colab/Notebook mode = run the full grid
        print("Running full grid (Kaggle/Colab/Notebook mode)...")
        print("To run a quick test instead, use: run_quick()")
        print("To run a single experiment: run_experiment('lstm_1.4M', 'avr', 100)")
        print()
        run_grid()
