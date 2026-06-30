"""
V5: What Is AVR? — Comprehensive evaluation on LFM2.5-350M.

This experiment answers the fundamental questions:
  1. Does AVR-repr (V2/V3 style, hidden-state MSE) work on this model?
  2. Does AVR-loss (V4 style, loss-based detection) work on this model?
  3. Does conv-only repair work? Attn-only? Both?
  4. Does the verification gate matter?
  5. Does dream (loss-based consolidation) work?
  6. Is it fair vs EWC?

METHODS:
  naive           = standard LoRA fine-tuning (all layers), no CL
  avr_repr_all    = AVR-repr detection + repair all LoRA params
  avr_loss_all    = AVR-loss detection + repair all LoRA params
  avr_loss_conv   = AVR-loss detection + repair conv LoRA only
  avr_loss_attn   = AVR-loss detection + repair attn LoRA only
  avr_always_conv = AVR-loss NO GATE + repair conv LoRA (ablation)
  ewc             = Elastic Weight Consolidation on LoRA params
  dream           = AVR-loss + conv repair + dream step (attn joint training)

ARCHITECTURE: LFM2.5-350M (10 conv + 6 attn layers)
  Conv layers = [0,1,3,4,6,7,9,11,13,15] (fast, plastic)
  Attn layers = [2,5,8,10,12,14]         (slow, stable)

LoRA: rank 16, applied to BOTH conv and attn layers.
  Conv target modules: in_proj, out_proj
  Attn target modules: q_proj, v_proj

FIXES from previous versions:
  - V2 double-shift bug: NOT present (using chunk.clone() like V3)
  - V3 consolidation net-negative: dream uses LOSS not MSE, joint not single-domain
  - V4 hidden-state too robust: both repr AND loss anchors tested here
  - EWC Fisher from full phase: using full phase here (pretrained model, full-phase training)
  - Results always saved as JSON

Kaggle/Colab: just hit Run All. Or:
  run_grid()                    # Full grid (~4-6 hrs on T4)
  run_quick()                   # Quick test: naive + avr_loss_conv (~30 min)
  run_experiment("avr_loss_conv")  # Single method
"""

import os, json, time, random, math, argparse, gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# ARCHITECTURE CONSTANTS
# ──────────────────────────────────────────────

# LFM2.5-350M has 16 layers: 10 conv (fast) + 6 attention (slow)
CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]

# Hidden state indices (layer_idx + 1 because hidden_states[0] = embedding)
CONV_HIDDEN_IDS = [idx + 1 for idx in CONV_LAYER_IDS]
ATTN_HIDDEN_IDS = [idx + 1 for idx in ATTN_LAYER_IDS]
ALL_HIDDEN_IDS = CONV_HIDDEN_IDS + ATTN_HIDDEN_IDS


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

@dataclass
class ModelConfig:
    hf_id: str = "LiquidAI/LFM2.5-350M"
    context_length: int = 512
    batch_size: int = 8          # Conservative for T4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    conv_targets: list = field(default_factory=lambda: ["in_proj", "out_proj"])
    attn_targets: list = field(default_factory=lambda: ["q_proj", "v_proj"])

@dataclass
class DomainConfig:
    name: str = ""
    display_name: str = ""
    dataset_name: str = ""
    text_field: str = "text"
    split: str = "train"
    max_tokens: int = 1_000_000  # 1M per domain — fits in T4 memory

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs_per_phase: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_samples: int = 1024
    results_dir: str = "v5_results"

@dataclass
class AVRConfig:
    n_anchor_probes: int = 50
    # AVR-repr (hidden-state MSE) threshold
    repr_drift_threshold: float = 0.1
    # AVR-loss (loss-based) threshold — 5% loss increase = degraded
    loss_drift_threshold: float = 0.05
    repair_steps: int = 50
    repair_lr: float = 1e-4
    verify_every_n_steps: int = 100  # Verify every N training steps

@dataclass
class DreamConfig:
    dream_steps: int = 100
    dream_lr: float = 1e-5        # Very low LR for dream
    dream_n_probes: int = 50     # Reduced from 200 — 65536 vocab makes logit tensor huge
    dream_stop_patience: int = 3  # Stop if old domain loss increases N times
    dream_batch_size: int = 2     # Mini-batch for dream to avoid OOM (was 4, OOM'd on T4)

@dataclass
class EWCConfig:
    lambda_: float = 0.1
    fisher_n_samples: int = 512


MODEL_CONFIG = ModelConfig()

DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text", max_tokens=1_000_000),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output", max_tokens=1_000_000),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text", max_tokens=1_000_000),
}

METHODS = ["naive", "avr_repr_all", "avr_loss_all", "avr_loss_conv", "avr_loss_attn",
           "avr_always_conv", "ewc", "dream"]


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    """Tokenized text dataset with non-overlapping chunks."""

    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, len(token_ids) // context_length)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length
        chunk = self.token_ids[start:end]
        # V3-style: chunk.clone() for labels. The model handles the shift internally.
        # This avoids the V2 double-shift bug.
        return {"input_ids": chunk, "labels": chunk.clone()}


def prepare_domain(domain, tokenizer, context_length, max_tokens, seed=42):
    from datasets import load_dataset
    print(f"  Loading: {domain.display_name}")
    ds = load_dataset(path=domain.dataset_name, split=domain.split)
    texts = [t for t in ds[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed)
    random.shuffle(texts)

    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens:
            break

    token_ids = torch.tensor(all_tokens[:int(max_tokens)], dtype=torch.long)
    print(f"    {len(token_ids):,} tokens")

    n_val = min(int(len(token_ids) * 0.1), 100_000)
    n_train = len(token_ids) - n_val

    train_ds = TextDataset(token_ids[:n_train], context_length)
    val_ds = TextDataset(token_ids[n_train:n_train + n_val], context_length)
    return train_ds, val_ds


# ──────────────────────────────────────────────
# MODEL SETUP
# ──────────────────────────────────────────────

def create_model(mcfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  Loading {mcfg.hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id,
        trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    # Apply LoRA to BOTH conv and attn layers
    all_targets = list(set(mcfg.conv_targets + mcfg.attn_targets))
    lora_config = LoraConfig(
        r=mcfg.lora_rank,
        lora_alpha=mcfg.lora_alpha,
        lora_dropout=mcfg.lora_dropout,
        target_modules=all_targets,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def is_conv_param(name):
    """Check if a parameter belongs to a conv layer."""
    for idx in CONV_LAYER_IDS:
        if f"layers.{idx}." in name:
            return True
    return False

def is_attn_param(name):
    """Check if a parameter belongs to an attention layer."""
    for idx in ATTN_LAYER_IDS:
        if f"layers.{idx}." in name:
            return True
    return False

def get_param_group(model, layer_type="all"):
    """Get trainable parameters filtered by layer type."""
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            continue
        if layer_type == "conv" and not is_conv_param(name):
            continue
        if layer_type == "attn" and not is_attn_param(name):
            continue
        params.append((name, param))
    return params

def freeze_lora(model, layer_type):
    """Freeze LoRA params for a specific layer type."""
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name:
            if layer_type == "conv" and is_conv_param(name):
                param.requires_grad = False
            elif layer_type == "attn" and is_attn_param(name):
                param.requires_grad = False

def unfreeze_lora(model, layer_type):
    """Unfreeze LoRA params for a specific layer type."""
    for name, param in model.named_parameters():
        if "lora_" in name:
            if layer_type == "conv" and is_conv_param(name):
                param.requires_grad = True
            elif layer_type == "attn" and is_attn_param(name):
                param.requires_grad = True


# ──────────────────────────────────────────────
# ANCHOR STORES (both repr and loss)
# ──────────────────────────────────────────────

class ReprAnchorStore:
    """AVR-repr: Hidden-state MSE anchors (V2/V3 style)."""

    def __init__(self):
        self.anchors = {}
        self.health = {}

    def save_anchors(self, model, phase_key, dataset, n_probes, device):
        model.eval()
        n_avail = len(dataset)
        n_probes = min(n_probes, n_avail)
        indices = random.sample(range(n_avail), n_probes)
        probe_ids = torch.stack([dataset[i]["input_ids"] for i in indices]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=probe_ids, output_hidden_states=True)

        hidden = {}
        for idx, hs in enumerate(outputs.hidden_states):
            if idx in ALL_HIDDEN_IDS:
                hidden[idx] = hs.mean(dim=1).cpu()

        self.anchors[phase_key] = {"probes": probe_ids.cpu(), "hidden": hidden, "n": n_probes}
        self.health[phase_key] = 1.0
        kb = sum(v.numel() for v in hidden.values()) * 4 / 1024 + probe_ids.numel() * 4 / 1024
        print(f"  Repr Anchors: Phase {phase_key} ({len(hidden)} layers, ~{kb:.1f}KB)")
        model.train()

    def verify(self, model, phase_keys, threshold, device):
        drift_report, degraded = {}, []
        for pk in phase_keys:
            if pk not in self.anchors:
                continue
            probes = self.anchors[pk]["probes"].to(device)
            with torch.no_grad():
                outputs = model(input_ids=probes, output_hidden_states=True)

            pdrift = {}
            for idx, hs in enumerate(outputs.hidden_states):
                if idx not in self.anchors[pk]["hidden"]:
                    continue
                pdrift[idx] = F.mse_loss(hs.mean(dim=1), self.anchors[pk]["hidden"][idx].to(device)).item()

            drift_report[pk] = pdrift
            if pdrift and max(pdrift.values()) > threshold:
                degraded.append(pk)
            avg = sum(pdrift.values()) / max(len(pdrift), 1)
            self.health[pk] = max(0.0, 1.0 - avg / (threshold * 5))
        return drift_report, degraded, len(degraded) > 0


class LossAnchorStore:
    """AVR-loss: Loss-based anchors (V4 style)."""

    def __init__(self):
        self.anchors = {}
        self.health = {}

    def save_anchors(self, model, phase_key, dataset, n_probes, device):
        model.train()  # cuDNN compat
        n_avail = len(dataset)
        n_probes = min(n_probes, n_avail)
        indices = random.sample(range(n_avail), n_probes)
        probe_ids = torch.stack([dataset[i]["input_ids"] for i in indices])
        labels = torch.stack([dataset[i]["labels"] for i in indices])

        # Mini-batch to avoid OOM: 65536 vocab × 512 ctx × N probes = huge logit tensor
        # Process in chunks of 4 to keep memory under control
        ANCHOR_BS = 4
        total_loss, total_tok, n_batches = 0.0, 0, 0
        with torch.no_grad():
            for i in range(0, n_probes, ANCHOR_BS):
                batch_ids = probe_ids[i:i+ANCHOR_BS].to(device)
                batch_labels = labels[i:i+ANCHOR_BS].to(device)
                outputs = model(input_ids=batch_ids, labels=batch_labels)
                nt = batch_labels.numel()
                total_loss += outputs.loss.item() * nt
                total_tok += nt
                n_batches += 1
                del outputs
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        initial_loss = total_loss / total_tok if total_tok > 0 else 0.0

        self.anchors[phase_key] = {
            "probes": probe_ids, "labels": labels,
            "initial_loss": initial_loss, "n": n_probes
        }
        self.health[phase_key] = 1.0
        print(f"  Loss Anchors: Phase {phase_key} (loss={initial_loss:.4f}, {n_probes} probes)")

    def verify(self, model, phase_keys, threshold, device):
        drift_report, degraded = {}, []
        ANCHOR_BS = 4  # Same mini-batching as save_anchors
        for pk in phase_keys:
            if pk not in self.anchors:
                continue
            probes = self.anchors[pk]["probes"]
            labels = self.anchors[pk]["labels"]
            n_probes = self.anchors[pk]["n"]
            # Mini-batch to avoid OOM
            total_loss, total_tok, n_batches = 0.0, 0, 0
            with torch.no_grad():
                for i in range(0, n_probes, ANCHOR_BS):
                    batch_ids = probes[i:i+ANCHOR_BS].to(device)
                    batch_labels = labels[i:i+ANCHOR_BS].to(device)
                    outputs = model(input_ids=batch_ids, labels=batch_labels)
                    nt = batch_labels.numel()
                    total_loss += outputs.loss.item() * nt
                    total_tok += nt
                    n_batches += 1
                    del outputs
            current_loss = total_loss / total_tok if total_tok > 0 else 0.0
            initial_loss = self.anchors[pk]["initial_loss"]
            rel_increase = (current_loss - initial_loss) / initial_loss if initial_loss > 0 else 0.0
            drift_report[pk] = {"initial_loss": initial_loss, "current_loss": current_loss, "rel_increase": rel_increase}
            self.health[pk] = max(0.0, 1.0 - rel_increase / threshold)
            if rel_increase > threshold:
                degraded.append(pk)
        return drift_report, degraded, len(degraded) > 0


# ──────────────────────────────────────────────
# METHODS
# ──────────────────────────────────────────────

class NaiveMethod:
    def __init__(self):
        self.name = "naive"
        self.completed = []
        self.extra_steps = 0

    def on_phase_start(self, model, pk):
        # All LoRA trainable
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device):
        pass

    def compute_loss(self, model, batch, device):
        outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return outputs.loss, {"lm_loss": outputs.loss.item()}


class AVRMethod:
    """AVR with configurable detection (repr/loss) and repair target (conv/attn/all)."""

    def __init__(self, detection="loss", repair_target="all", gated=True):
        self.detection = detection  # "repr" or "loss"
        self.repair_target = repair_target  # "conv", "attn", or "all"
        self.gated = gated

        self.cfg = AVRConfig()
        self.name = f"avr_{detection}_{repair_target}" + ("" if gated else "_always")

        if detection == "repr":
            self.anchor_store = ReprAnchorStore()
        else:
            self.anchor_store = LossAnchorStore()

        self.snapshots = {}
        self.completed = []
        self.extra_steps = 0
        self.total_repairs = 0
        self.total_verifies = 0
        self.step_count = 0

    def on_phase_start(self, model, pk):
        self.step_count = 0
        # All LoRA trainable during absorption
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        # Save anchors
        self.anchor_store.save_anchors(model, pk, dataset, self.cfg.n_anchor_probes, device)
        # Save weight snapshot for repair
        snap = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name:
                snap[name] = param.data.cpu().clone()
        self.snapshots[pk] = snap
        n_params = sum(v.numel() for v in snap.values())
        print(f"  Snapshot: Phase {pk} ({len(snap)} params, ~{n_params*4/1024:.1f}KB)")
        # Verify + repair
        self._verify_repair(model, device)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device):
        self.step_count = step
        if self.completed and step % self.cfg.verify_every_n_steps == 0 and step > 0:
            if self.gated:
                self._verify_repair(model, device)
            else:
                self._always_repair(model, device)

    def compute_loss(self, model, batch, device):
        outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return outputs.loss, {"lm_loss": outputs.loss.item()}

    def _verify_repair(self, model, device):
        self.total_verifies += 1
        threshold = self.cfg.repr_drift_threshold if self.detection == "repr" else self.cfg.loss_drift_threshold
        drift_report, degraded, needs = self.anchor_store.verify(model, self.completed, threshold, device)

        for pk in self.completed:
            h = self.anchor_store.health.get(pk, 1.0)
            if self.detection == "loss":
                dr = drift_report.get(pk, {})
                rel = dr.get("rel_increase", 0)
                print(f"  [VERIFY] {pk}: loss {dr.get('initial_loss',0):.3f}→{dr.get('current_loss',0):.3f} (+{rel*100:.1f}%) health={h:.3f} [{'OK' if h > 0 else 'DEGRADED'}]")
            else:
                print(f"  [VERIFY] {pk}: health={h:.3f} [{'OK' if h > 0.5 else 'DEGRADED'}]")

        if not needs:
            return

        self._run_repair(model, degraded, device)

    def _always_repair(self, model, device):
        """Ablation: always repair, no verification gate."""
        if not self.snapshots:
            return
        self.total_repairs += 1
        self.total_verifies += 1
        degraded = list(self.completed)
        print(f"  [REPAIR-ALWAYS] {degraded} — {self.cfg.repair_steps} steps (no gate)")
        self._run_repair(model, degraded, device)

    def _run_repair(self, model, degraded, device):
        self.total_repairs += 1
        # Filter parameters by repair target
        repair_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" not in name:
                continue
            if self.repair_target == "conv" and not is_conv_param(name):
                continue
            if self.repair_target == "attn" and not is_attn_param(name):
                continue
            repair_params.append((name, param))

        if not repair_params:
            print(f"  [REPAIR] No {self.repair_target} LoRA params to repair")
            return

        print(f"  [REPAIR] {degraded} — {self.cfg.repair_steps} steps ({self.repair_target} LoRA, {len(repair_params)} params)")
        trainable = [p for _, p in repair_params]
        opt = torch.optim.Adam(trainable, lr=self.cfg.repair_lr)

        for _ in range(self.cfg.repair_steps):
            wl = torch.tensor(0.0, device=device)
            n = 0
            for pk in set(degraded):
                if pk not in self.snapshots:
                    continue
                for name, param in repair_params:
                    if name in self.snapshots[pk]:
                        wl = wl + F.mse_loss(param, self.snapshots[pk][name].to(device))
                        n += 1
            if n > 0 and wl.requires_grad:
                opt.zero_grad()
                (wl / n).backward()
                opt.step()
                self.extra_steps += 1
            else:
                break

        # Re-verify
        threshold = self.cfg.repr_drift_threshold if self.detection == "repr" else self.cfg.loss_drift_threshold
        _, new_deg, _ = self.anchor_store.verify(model, self.completed, threshold, device)
        print(f"  [REPAIR] {'OK' if not new_deg else f'Partial: {new_deg}'}")


class EWCMethod:
    def __init__(self):
        self.cfg = EWCConfig()
        self.name = "ewc"
        self.completed = []
        self.extra_steps = 0
        self.fisher = {}
        self.opt_params = {}
        self.computable = {}

    def on_phase_start(self, model, pk):
        pass

    def on_phase_end(self, model, pk, dataset, device):
        model.train()
        n_avail = len(dataset)
        if n_avail < 50:
            print(f"  [EWC] Only {n_avail} examples — Fisher UNRELIABLE")
            self.computable[pk] = False
            self.completed.append(pk)
            return

        self.computable[pk] = True
        n_samp = min(self.cfg.fisher_n_samples, n_avail)
        print(f"  [EWC] Computing Fisher from {n_samp} samples...")

        f_dict = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if p.requires_grad}
        o_dict = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        done = 0
        for batch in loader:
            if done >= n_samp:
                break
            model.zero_grad()
            outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            outputs.loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    f_dict[n] += p.grad.data.pow(2) / n_samp
            done += batch["input_ids"].size(0)

        self.fisher[pk] = f_dict
        self.opt_params[pk] = o_dict
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device):
        pass

    def compute_loss(self, model, batch, device):
        outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        lm = outputs.loss
        ewc = torch.tensor(0.0, device=device)
        for pk in self.completed:
            if not self.computable.get(pk, False):
                continue
            for n, p in model.named_parameters():
                if p.requires_grad and n in self.fisher.get(pk, {}):
                    ewc = ewc + (self.fisher[pk][n].to(device) * (p - self.opt_params[pk][n].to(device)).pow(2)).sum()
        return lm + self.cfg.lambda_ * ewc, {"lm_loss": lm.item(), "ewc_loss": ewc.item()}


class DreamMethod(AVRMethod):
    """
    AVR-loss + conv repair + dream step.

    After ABSORB+VERIFY+REPAIR:
      - Freeze conv LoRA
      - Train attn LoRA on JOINT loss over probes from ALL completed domains
      - If any old domain's loss increases during dream → stop (safety brake)
      - This replaces V3's MSE distillation with direct loss minimization
    """

    def __init__(self):
        super().__init__(detection="loss", repair_target="conv", gated=True)
        self.name = "dream"
        self.dream_cfg = DreamConfig()
        self.dream_probes = {}  # phase_key -> {probes, labels}

    def on_phase_end(self, model, pk, dataset, device):
        # Standard AVR anchors + snapshot
        self.anchor_store.save_anchors(model, pk, dataset, self.cfg.n_anchor_probes, device)
        snap = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_" in name:
                snap[name] = param.data.cpu().clone()
        self.snapshots[pk] = snap

        # Save extra probes for dream
        n_dream = min(self.dream_cfg.dream_n_probes, len(dataset))
        indices = random.sample(range(len(dataset)), n_dream)
        dream_probes = torch.stack([dataset[i]["input_ids"] for i in indices])
        dream_labels = torch.stack([dataset[i]["labels"] for i in indices])
        self.dream_probes[pk] = {"probes": dream_probes, "labels": dream_labels, "n": n_dream}
        print(f"  Dream probes: Phase {pk} ({n_dream} probes)")

        # Verify + repair (conv only)
        self._verify_repair(model, device)
        self.completed.append(pk)

        # Run dream
        self._dream(model, device)

    def _dream(self, model, device):
        """Train attn LoRA on joint loss from ALL completed domains."""
        if len(self.completed) < 2:
            print(f"  [DREAM] Skipped (only {len(self.completed)} domain)")
            return

        # Free memory before dream — anchors/repair leave fragmentation
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"  [DREAM] Training attn LoRA on {len(self.completed)} domains ({self.dream_cfg.dream_steps} steps)")

        DREAM_BS = self.dream_cfg.dream_batch_size

        # Compute baseline losses for safety brake (mini-batched)
        baseline_losses = {}
        for pk in self.completed:
            probes = self.dream_probes[pk]["probes"]
            labels = self.dream_probes[pk]["labels"]
            n = self.dream_probes[pk]["n"]
            total_loss, total_tok = 0.0, 0
            with torch.no_grad():
                for i in range(0, n, DREAM_BS):
                    batch_ids = probes[i:i+DREAM_BS].to(device)
                    batch_labels = labels[i:i+DREAM_BS].to(device)
                    outputs = model(input_ids=batch_ids, labels=batch_labels)
                    nt = batch_labels.numel()
                    total_loss += outputs.loss.item() * nt
                    total_tok += nt
                    del outputs
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
            baseline_losses[pk] = total_loss / total_tok if total_tok > 0 else 0.0

        # Freeze conv, unfreeze attn
        freeze_lora(model, "conv")
        unfreeze_lora(model, "attn")

        attn_params = [(n, p) for n, p in model.named_parameters()
                       if p.requires_grad and "lora_" in n and is_attn_param(n)]
        if not attn_params:
            print(f"  [DREAM] No attn LoRA params — skipping")
            unfreeze_lora(model, "conv")
            freeze_lora(model, "attn")
            return

        trainable = [p for _, p in attn_params]
        opt = torch.optim.Adam(trainable, lr=self.dream_cfg.dream_lr)
        n_violations = 0

        for step in range(self.dream_cfg.dream_steps):
            # Joint loss over all completed domains (mini-batched)
            joint_loss = torch.tensor(0.0, device=device)
            n_domains = 0
            for pk in self.completed:
                probes = self.dream_probes[pk]["probes"]
                labels = self.dream_probes[pk]["labels"]
                n = self.dream_probes[pk]["n"]
                domain_loss = torch.tensor(0.0, device=device)
                domain_tok = 0
                for i in range(0, n, DREAM_BS):
                    batch_ids = probes[i:i+DREAM_BS].to(device)
                    batch_labels = labels[i:i+DREAM_BS].to(device)
                    outputs = model(input_ids=batch_ids, labels=batch_labels)
                    domain_loss = domain_loss + outputs.loss * batch_labels.numel()
                    domain_tok += batch_labels.numel()
                if domain_tok > 0:
                    joint_loss = joint_loss + domain_loss / domain_tok
                    n_domains += 1

            if n_domains > 0:
                joint_loss = joint_loss / n_domains
                opt.zero_grad()
                joint_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                self.extra_steps += 1

            # Safety brake: check if any old domain got worse (mini-batched)
            if step % 10 == 0:
                violated = False
                for pk in self.completed:
                    probes = self.dream_probes[pk]["probes"]
                    labels = self.dream_probes[pk]["labels"]
                    n = self.dream_probes[pk]["n"]
                    total_loss, total_tok = 0.0, 0
                    with torch.no_grad():
                        for i in range(0, n, DREAM_BS):
                            batch_ids = probes[i:i+DREAM_BS].to(device)
                            batch_labels = labels[i:i+DREAM_BS].to(device)
                            outputs = model(input_ids=batch_ids, labels=batch_labels)
                            nt = batch_labels.numel()
                            total_loss += outputs.loss.item() * nt
                            total_tok += nt
                            del outputs
                    current_loss = total_loss / total_tok if total_tok > 0 else 0.0
                    if current_loss > baseline_losses[pk] * 1.1:  # 10% tolerance
                        violated = True
                        break

                if violated:
                    n_violations += 1
                    if n_violations >= self.dream_cfg.dream_stop_patience:
                        print(f"  [DREAM] STOPPED at step {step} — old domain loss increasing")
                        break
                else:
                    n_violations = 0

            if step % 25 == 0:
                print(f"    Dream step {step}/{self.dream_cfg.dream_steps} | joint_loss={joint_loss.item():.4f}")

        # Restore: unfreeze conv, freeze attn
        unfreeze_lora(model, "conv")
        freeze_lora(model, "attn")
        print(f"  [DREAM] Done ({self.extra_steps} total extra steps)")


# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, dataset, device, max_samp=1024):
    model.eval()
    eval_bs = 8
    loader = DataLoader(dataset, batch_size=eval_bs, shuffle=False)
    tot_loss, tot_tok, nb = 0.0, 0, 0
    for batch in loader:
        if nb * eval_bs >= max_samp:
            break
        outputs = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        nt = batch["labels"].numel()  # All tokens count (no padding with pretrained tokenizer)
        tot_loss += outputs.loss.item() * nt
        tot_tok += nt
        nb += 1
    model.train()
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")


def eval_all(model, val_ds, phases, device, max_samp=1024):
    return {"perplexity": {pk: compute_ppl(model, val_ds[pk], device, max_samp) for pk in phases if pk in val_ds}}


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

def run_experiment(method_name, seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

    tc = TrainConfig()
    mcfg = MODEL_CONFIG
    device = tc.device

    print(f"\n{'#'*60}")
    print(f"# {method_name} | device={device}")
    print(f"{'#'*60}")

    # Create model
    model, tokenizer = create_model(mcfg, device)

    # Prepare data
    phases_data = {}
    val_ds = {}
    for pk, d in DOMAINS.items():
        train_ds, val = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = train_ds
        val_ds[pk] = val

    # Create method
    if method_name == "naive":
        method = NaiveMethod()
    elif method_name == "avr_repr_all":
        method = AVRMethod(detection="repr", repair_target="all", gated=True)
    elif method_name == "avr_loss_all":
        method = AVRMethod(detection="loss", repair_target="all", gated=True)
    elif method_name == "avr_loss_conv":
        method = AVRMethod(detection="loss", repair_target="conv", gated=True)
    elif method_name == "avr_loss_attn":
        method = AVRMethod(detection="loss", repair_target="attn", gated=True)
    elif method_name == "avr_always_conv":
        method = AVRMethod(detection="loss", repair_target="conv", gated=False)
    elif method_name == "ewc":
        method = EWCMethod()
    elif method_name == "dream":
        method = DreamMethod()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    results = {"method": method_name, "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        dataset = phases_data[pk]
        method.on_phase_start(model, pk)

        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)

        gs, tl = 0, 0.0
        t0 = time.time()
        print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} ({len(dataset)} samples)")

        for epoch in range(tc.epochs_per_phase):
            for batch in loader:
                model.train()
                loss, metrics = method.compute_loss(model, batch, device)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step()
                tl += loss.item()
                gs += 1

                method.on_step_end(model, pk, gs, device)

                if gs % 100 == 0:
                    print(f"    step {gs} | avg_loss={tl/max(gs,1):.4f}")

        method.on_phase_end(model, pk, dataset, device)
        model.train()  # Safety

        ev = eval_all(model, val_ds, method.completed if hasattr(method, 'completed') else [pk], device, tc.eval_samples)
        ev["avg_loss"] = tl / max(gs, 1)
        ev["extra_steps"] = method.extra_steps if hasattr(method, 'extra_steps') else 0
        if hasattr(method, 'total_repairs'): ev["repairs"] = method.total_repairs
        if hasattr(method, 'total_verifies'): ev["verifies"] = method.total_verifies
        if hasattr(method, 'computable'): ev["ewc_computable"] = method.computable
        results["phases"][pk] = ev
        ev["time"] = time.time() - t0

        print(f"  Eval:")
        for p, ppl in ev.get("perplexity", {}).items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save
    os.makedirs(tc.results_dir, exist_ok=True)
    fp = os.path.join(tc.results_dir, f"{method_name}.json")
    with open(fp, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {fp}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results


# ──────────────────────────────────────────────
# GRID RUNNER
# ──────────────────────────────────────────────

def run_grid(methods=None):
    methods = methods or METHODS
    all_res = []
    # Skip methods that already have results
    os.makedirs("v5_results", exist_ok=True)
    completed = [f.replace(".json", "") for f in os.listdir("v5_results") if f.endswith(".json") and f != "full_grid.json"]
    if completed:
        print(f"  Already completed: {completed}")

    for meth in methods:
        if meth in completed:
            print(f"\n{'*'*60}")
            print(f"  SKIP {meth} (already completed)")
            fp = os.path.join("v5_results", f"{meth}.json")
            with open(fp) as f:
                all_res.append(json.load(f))
            continue

        print(f"\n{'*'*60}")
        try:
            r = run_experiment(meth)
            all_res.append(r)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    with open("v5_results/full_grid.json", "w") as f:
        json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res


def run_quick():
    """Quick test: naive + avr_loss_conv. ~1 hour."""
    print("QUICK TEST: naive + avr_loss_conv")
    return run_grid(["naive", "avr_loss_conv"])


def print_summary(results=None):
    if results is None:
        fps = [os.path.join("v5_results", f) for f in os.listdir("v5_results")
               if f.endswith(".json") and f != "full_grid.json"]
        results = []
        for fp in fps:
            with open(fp) as f:
                results.append(json.load(f))
        fg = os.path.join("v5_results", "full_grid.json")
        if os.path.exists(fg):
            with open(fg) as f:
                results.extend(json.load(f))

    if not results:
        print("No results found")
        return

    print(f"\n{'='*120}")
    print(f"V5: WHAT IS AVR? — RESULTS ON LFM2.5-350M")
    print(f"{'='*120}")
    print(f"{'Method':<20} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<8} {'FF(B)':<8} {'Repairs':<8} {'ExtraSt':<8}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x.get("method", "")):
        method = r.get("method", "?")
        phases = r.get("phases", {})
        keys = list(phases.keys())

        if len(keys) < 3:
            print(f"{method:<20} (incomplete)")
            continue

        # Final PPLs (after all 3 phases)
        final_ppl = phases[keys[-1]].get("perplexity", {})
        a_ppl = final_ppl.get("A", 0)
        b_ppl = final_ppl.get("B", 0)
        c_ppl = final_ppl.get("C", 0)

        # Forgetting factors: PPL after C / PPL right after learning
        ff_a = a_ppl / phases["A"].get("perplexity", {}).get("A", 1) if a_ppl > 0 else 0
        ff_b = b_ppl / phases["B"].get("perplexity", {}).get("B", 1) if b_ppl > 0 else 0

        lp = phases[keys[-1]] if keys else {}
        repairs = lp.get("repairs", "-")
        extra = lp.get("extra_steps", "-")

        print(f"{method:<20} {a_ppl:<10.1f} {b_ppl:<10.1f} {c_ppl:<10.1f} {ff_a:<8.2f}x {ff_b:<8.2f}x {repairs:<8} {extra:<8}")

    print(f"{'='*120}")
    print(f"FF(X) = PPL on X after all training / PPL on X right after learning X")
    print(f"Lower FF = less forgetting. PPL columns = absolute PPL after all training (lower = better)")


# ──────────────────────────────────────────────
# CLI / KAGGLE ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="V5: What Is AVR?")
    parser.add_argument("--method", default=None, choices=METHODS)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.quick:
        run_quick()
    elif args.grid:
        run_grid()
    elif args.summary:
        print_summary()
    elif args.method:
        run_experiment(args.method)
    else:
        # Default: run full grid
        print("Running full grid (Kaggle/Colab/Notebook mode)...")
        print("Quick test: run_quick()")
        print("Single method: run_experiment('avr_loss_conv')")
        print()
        run_grid()
