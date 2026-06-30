"""
V6: The Living Model — Accumulate, Don't Consolidate.

Core idea: Instead of moving knowledge from conv->attn (which keeps failing
because shared attn LoRA overwrites previous domains), MERGE conv LoRA
into base weights after each domain. This permanently "commits" knowledge,
freeing conv LoRA capacity for the next domain.

Why this might work on LFM2.5 but NOT on transformers:
  Conv layers are position-local -> merging doesn't create cross-domain interference
  Attn layers are position-global -> merging WOULD create interference

Cycle per domain:
  ABSORB:  Train conv LoRA on new domain (attn frozen or shared)
  VERIFY:  Check loss on old domain probes (loss-based, 5% threshold)
  REPAIR:  Pull conv LoRA toward snapshot if degraded
  COMMIT:  Merge conv LoRA into base weights. Zero conv LoRA. Ready for next domain.

The base weights ACCUMULATE all domain knowledge. The conv LoRA always has
full capacity for the new domain. No consolidation needed.

METHODS:
  naive               = standard LoRA, no CL (V5 baseline)
  accumulate          = conv LoRA only, merge into base after each domain
  accumulate_avr      = same + verify/repair before merge
  accumulate_shared   = conv accumulate + shared attn LoRA

ARCHITECTURE: LFM2.5-350M (10 conv + 6 attn layers)
"""

import os, json, time, random, math, argparse, gc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]

@dataclass
class ModelConfig:
    hf_id: str = "LiquidAI/LFM2.5-350M"
    context_length: int = 512
    batch_size: int = 8
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
    max_tokens: int = 1_000_000

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs_per_phase: int = 1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_samples: int = 1024
    results_dir: str = "v6_results"

@dataclass
class AVRConfig:
    n_anchor_probes: int = 50
    loss_drift_threshold: float = 0.05
    repair_steps: int = 50
    repair_lr: float = 1e-4
    verify_every_n_steps: int = 100

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text"),
}
METHODS = ["naive", "accumulate", "accumulate_avr", "accumulate_shared"]


# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, token_ids, context_length):
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, len(token_ids) // context_length)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.context_length
        e = s + self.context_length
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}

def prepare_domain(domain, tokenizer, context_length, max_tokens, seed=42):
    from datasets import load_dataset
    print(f"  Loading: {domain.display_name}")
    ds = load_dataset(path=domain.dataset_name, split=domain.split)
    texts = [t for t in ds[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed); random.shuffle(texts)
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens: break
    token_ids = torch.tensor(all_tokens[:int(max_tokens)], dtype=torch.long)
    print(f"    {len(token_ids):,} tokens")
    n_val = min(int(len(token_ids) * 0.1), 100_000)
    n_train = len(token_ids) - n_val
    return TextDataset(token_ids[:n_train], context_length), TextDataset(token_ids[n_train:n_train + n_val], context_length)


# ──────────────────────────────────────────────
# MODEL + LoRA HELPERS
# ──────────────────────────────────────────────

def is_conv_module(name):
    for idx in CONV_LAYER_IDS:
        if f"layers.{idx}." in name: return True
    return False

def is_attn_module(name):
    for idx in ATTN_LAYER_IDS:
        if f"layers.{idx}." in name: return True
    return False

def _load_base(mcfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {mcfg.hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id, trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    return model, tokenizer

def create_model_conv_only(mcfg, device):
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(mcfg, device)
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=mcfg.conv_targets, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer

def create_model_all_lora(mcfg, device):
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(mcfg, device)
    all_targets = list(set(mcfg.conv_targets + mcfg.attn_targets))
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=all_targets, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def merge_conv_lora_into_base(model):
    """COMMIT: Merge conv-layer LoRA into base weights, then re-init for next domain.
    
    After this:
    - Base weights = original + all previous conv LoRA deltas (accumulated)
    - Conv LoRA A = kaiming init, B = zeros (standard LoRA init, ready to learn)
    - Forward pass is identical: base_weight + (B @ A) * scaling = base_weight + 0
    """
    from peft.tuners.lora.layer import LoraLayer
    merged_count = 0
    adapter_name = "default"

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if adapter_name not in module.lora_A: continue

        with torch.no_grad():
            A = module.lora_A[adapter_name].weight.data
            B = module.lora_B[adapter_name].weight.data
            scaling = module.scaling[adapter_name]
            delta = (B @ A) * scaling
            module.base_layer.weight.data += delta
            # Re-init: A with kaiming (like fresh LoRA), B stays zero
            nn.init.kaiming_uniform_(module.lora_A[adapter_name].weight, a=math.sqrt(5))
            module.lora_B[adapter_name].weight.data.zero_()

        # Keep out of merged_adapters so forward uses LoRA path
        if adapter_name in module.merged_adapters:
            module.merged_adapters.remove(adapter_name)
        merged_count += 1

    print(f"  [COMMIT] Merged {merged_count} conv LoRA into base, re-init for next domain")
    return merged_count


# ──────────────────────────────────────────────
# LOSS ANCHOR STORE
# ──────────────────────────────────────────────

class LossAnchorStore:
    def __init__(self):
        self.anchors = {}
        self.health = {}

    def save_anchors(self, model, phase_key, dataset, n_probes, device):
        model.train()
        n_avail = len(dataset)
        n_probes = min(n_probes, n_avail)
        indices = random.sample(range(n_avail), n_probes)
        probe_ids = torch.stack([dataset[i]["input_ids"] for i in indices])
        labels = torch.stack([dataset[i]["labels"] for i in indices])
        BS = 4
        total_loss, total_tok = 0.0, 0
        with torch.no_grad():
            for i in range(0, n_probes, BS):
                bids = probe_ids[i:i+BS].to(device)
                blabs = labels[i:i+BS].to(device)
                out = model(input_ids=bids, labels=blabs)
                nt = blabs.numel()
                total_loss += out.loss.item() * nt
                total_tok += nt
                del out
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        initial_loss = total_loss / total_tok if total_tok > 0 else 0.0
        self.anchors[phase_key] = {"probes": probe_ids, "labels": labels, "initial_loss": initial_loss, "n": n_probes}
        self.health[phase_key] = 1.0
        print(f"  Anchors: Phase {phase_key} (loss={initial_loss:.4f}, {n_probes} probes)")

    def verify(self, model, phase_keys, threshold, device):
        drift_report, degraded = {}, []
        BS = 4
        for pk in phase_keys:
            if pk not in self.anchors: continue
            probes = self.anchors[pk]["probes"]
            labels = self.anchors[pk]["labels"]
            n = self.anchors[pk]["n"]
            total_loss, total_tok = 0.0, 0
            with torch.no_grad():
                for i in range(0, n, BS):
                    bids = probes[i:i+BS].to(device)
                    blabs = labels[i:i+BS].to(device)
                    out = model(input_ids=bids, labels=blabs)
                    nt = blabs.numel()
                    total_loss += out.loss.item() * nt
                    total_tok += nt
                    del out
            current_loss = total_loss / total_tok if total_tok > 0 else 0.0
            initial_loss = self.anchors[pk]["initial_loss"]
            rel = (current_loss - initial_loss) / initial_loss if initial_loss > 0 else 0.0
            drift_report[pk] = {"initial_loss": initial_loss, "current_loss": current_loss, "rel_increase": rel}
            self.health[pk] = max(0.0, 1.0 - rel / threshold)
            if rel > threshold: degraded.append(pk)
        return drift_report, degraded, len(degraded) > 0


# ──────────────────────────────────────────────
# METHODS
# ──────────────────────────────────────────────

class NaiveMethod:
    def __init__(self):
        self.name = "naive"
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True
    def on_phase_end(self, model, pk, dataset, device): self.completed.append(pk)
    def on_step_end(self, model, pk, step, device): pass
    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class AccumulateMethod:
    """Conv LoRA only. Merge into base after each domain. No verify/repair."""
    def __init__(self):
        self.name = "accumulate"
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True
    def on_phase_end(self, model, pk, dataset, device):
        merge_conv_lora_into_base(model)
        self.completed.append(pk)
    def on_step_end(self, model, pk, step, device): pass
    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class AccumulateAVRMethod:
    """Conv LoRA only. Verify + Repair before Commit."""
    def __init__(self):
        self.name = "accumulate_avr"
        self.cfg = AVRConfig()
        self.anchor_store = LossAnchorStore()
        self.snapshots = {}
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
        self.step_count = 0

    def on_phase_start(self, model, pk):
        self.step_count = 0
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        self.anchor_store.save_anchors(model, pk, dataset, self.cfg.n_anchor_probes, device)
        snap = {n: p.data.cpu().clone() for n, p in model.named_parameters()
                if p.requires_grad and "lora_" in n and is_conv_module(n)}
        self.snapshots[pk] = snap
        n_p = sum(v.numel() for v in snap.values())
        print(f"  Snapshot: Phase {pk} ({len(snap)} params, ~{n_p*4/1024:.1f}KB)")
        self._verify_repair(model, device)
        merge_conv_lora_into_base(model)
        # Old snapshots invalid after merge (base weights changed)
        self.snapshots.pop(pk, None)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device):
        self.step_count = step
        if self.completed and step % self.cfg.verify_every_n_steps == 0 and step > 0:
            self._verify_repair(model, device)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}

    def _verify_repair(self, model, device):
        self.total_verifies += 1
        drift, degraded, needs = self.anchor_store.verify(model, self.completed, self.cfg.loss_drift_threshold, device)
        for pk in self.completed:
            h = self.anchor_store.health.get(pk, 1.0)
            dr = drift.get(pk, {})
            rel = dr.get("rel_increase", 0)
            print(f"  [VERIFY] {pk}: loss {dr.get('initial_loss',0):.3f}->{dr.get('current_loss',0):.3f} (+{rel*100:.1f}%) health={h:.3f} [{'OK' if h > 0 else 'DEGRADED'}]")
        if not needs: return
        self.total_repairs += 1
        repair_params = [(n, p) for n, p in model.named_parameters()
                         if p.requires_grad and "lora_" in n and is_conv_module(n)]
        if not repair_params: return
        print(f"  [REPAIR] {degraded} - {self.cfg.repair_steps} steps (conv LoRA)")
        trainable = [p for _, p in repair_params]
        opt = torch.optim.Adam(trainable, lr=self.cfg.repair_lr)
        for _ in range(self.cfg.repair_steps):
            wl, n = torch.tensor(0.0, device=device), 0
            for pk in set(degraded):
                if pk not in self.snapshots: continue
                for name, param in repair_params:
                    if name in self.snapshots[pk]:
                        wl = wl + F.mse_loss(param, self.snapshots[pk][name].to(device))
                        n += 1
            if n > 0 and wl.requires_grad:
                opt.zero_grad(); (wl / n).backward(); opt.step(); self.extra_steps += 1
            else: break
        _, new_deg, _ = self.anchor_store.verify(model, self.completed, self.cfg.loss_drift_threshold, device)
        print(f"  [REPAIR] {'OK' if not new_deg else f'Partial: {new_deg}'}")


class AccumulateSharedMethod:
    """Conv LoRA (merge after each domain) + shared attn LoRA."""
    def __init__(self):
        self.name = "accumulate_shared"
        self.cfg = AVRConfig()
        self.anchor_store = LossAnchorStore()
        self.snapshots = {}
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
        self.step_count = 0

    def on_phase_start(self, model, pk):
        self.step_count = 0
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        self.anchor_store.save_anchors(model, pk, dataset, self.cfg.n_anchor_probes, device)
        snap = {n: p.data.cpu().clone() for n, p in model.named_parameters()
                if p.requires_grad and "lora_" in n and is_conv_module(n)}
        self.snapshots[pk] = snap
        self._verify_repair(model, device)
        merge_conv_lora_into_base(model)
        self.snapshots.pop(pk, None)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device):
        self.step_count = step
        if self.completed and step % self.cfg.verify_every_n_steps == 0 and step > 0:
            self._verify_repair(model, device)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}

    def _verify_repair(self, model, device):
        self.total_verifies += 1
        drift, degraded, needs = self.anchor_store.verify(model, self.completed, self.cfg.loss_drift_threshold, device)
        for pk in self.completed:
            h = self.anchor_store.health.get(pk, 1.0)
            dr = drift.get(pk, {})
            rel = dr.get("rel_increase", 0)
            print(f"  [VERIFY] {pk}: loss {dr.get('initial_loss',0):.3f}->{dr.get('current_loss',0):.3f} (+{rel*100:.1f}%) health={h:.3f} [{'OK' if h > 0 else 'DEGRADED'}]")
        if not needs: return
        self.total_repairs += 1
        # Repair conv LoRA only (attn is shared across domains)
        repair_params = [(n, p) for n, p in model.named_parameters()
                         if p.requires_grad and "lora_" in n and is_conv_module(n)]
        if not repair_params: return
        print(f"  [REPAIR] {degraded} - {self.cfg.repair_steps} steps (conv LoRA)")
        trainable = [p for _, p in repair_params]
        opt = torch.optim.Adam(trainable, lr=self.cfg.repair_lr)
        for _ in range(self.cfg.repair_steps):
            wl, n = torch.tensor(0.0, device=device), 0
            for pk in set(degraded):
                if pk not in self.snapshots: continue
                for name, param in repair_params:
                    if name in self.snapshots[pk]:
                        wl = wl + F.mse_loss(param, self.snapshots[pk][name].to(device))
                        n += 1
            if n > 0 and wl.requires_grad:
                opt.zero_grad(); (wl / n).backward(); opt.step(); self.extra_steps += 1
            else: break
        _, new_deg, _ = self.anchor_store.verify(model, self.completed, self.cfg.loss_drift_threshold, device)
        print(f"  [REPAIR] {'OK' if not new_deg else f'Partial: {new_deg}'}")


# ──────────────────────────────────────────────
# EVAL
# ──────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, dataset, device, max_samp=1024):
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    tot_loss, tot_tok, nb = 0.0, 0, 0
    for batch in loader:
        if nb * 8 >= max_samp: break
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        nt = batch["labels"].numel()
        tot_loss += out.loss.item() * nt; tot_tok += nt; nb += 1
    model.train()
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")

def eval_all(model, val_ds, phases, device, max_samp=1024):
    return {"perplexity": {pk: compute_ppl(model, val_ds[pk], device, max_samp) for pk in phases if pk in val_ds}}


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

def run_experiment(method_name, seed=42):
    torch.manual_seed(seed); random.seed(seed)
    tc = TrainConfig(); mcfg = MODEL_CONFIG; device = tc.device

    print(f"\n{'#'*60}")
    print(f"# {method_name} | device={device}")
    print(f"{'#'*60}")

    if method_name == "naive":
        model, tokenizer = create_model_all_lora(mcfg, device)
    elif method_name in ("accumulate", "accumulate_avr"):
        model, tokenizer = create_model_conv_only(mcfg, device)
    elif method_name == "accumulate_shared":
        model, tokenizer = create_model_all_lora(mcfg, device)
    else: raise ValueError(f"Unknown: {method_name}")

    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    if method_name == "naive": method = NaiveMethod()
    elif method_name == "accumulate": method = AccumulateMethod()
    elif method_name == "accumulate_avr": method = AccumulateAVRMethod()
    elif method_name == "accumulate_shared": method = AccumulateSharedMethod()

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
                loss, _ = method.compute_loss(model, batch, device)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += loss.item(); gs += 1
                method.on_step_end(model, pk, gs, device)
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/max(gs,1):.4f}")

        method.on_phase_end(model, pk, dataset, device)
        model.train()
        # Rebuild optimizer after merge (params changed)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)

        ev = eval_all(model, val_ds, method.completed, device, tc.eval_samples)
        ev["avg_loss"] = tl / max(gs, 1)
        ev["extra_steps"] = method.extra_steps
        ev["repairs"] = method.total_repairs
        ev["verifies"] = method.total_verifies
        results["phases"][pk] = ev
        ev["time"] = time.time() - t0
        print(f"  Eval:")
        for p, ppl in ev.get("perplexity", {}).items():
            print(f"    {p}: PPL={ppl:.2f}")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    os.makedirs(tc.results_dir, exist_ok=True)
    fp = os.path.join(tc.results_dir, f"{method_name}.json")
    with open(fp, "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {fp}")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ──────────────────────────────────────────────
# GRID
# ──────────────────────────────────────────────

def run_grid(methods=None):
    methods = methods or METHODS
    all_res = []
    os.makedirs("v6_results", exist_ok=True)
    done = [f.replace(".json","") for f in os.listdir("v6_results") if f.endswith(".json") and f != "full_grid.json"]
    if done: print(f"  Already done: {done}")
    for meth in methods:
        if meth in done:
            print(f"\n{'*'*60}\n  SKIP {meth}")
            with open(os.path.join("v6_results", f"{meth}.json")) as f: all_res.append(json.load(f))
            continue
        print(f"\n{'*'*60}")
        try: all_res.append(run_experiment(meth))
        except Exception as e: print(f"  FAILED: {e}"); import traceback; traceback.print_exc()
    with open("v6_results/full_grid.json", "w") as f: json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res

def run_quick():
    print("QUICK: naive + accumulate")
    return run_grid(["naive", "accumulate"])

def print_summary(results=None):
    if results is None:
        results = []
        for f in os.listdir("v6_results"):
            if f.endswith(".json") and f != "full_grid.json":
                with open(os.path.join("v6_results", f)) as fh: results.append(json.load(fh))
    if not results: print("No results"); return
    print(f"\n{'='*120}")
    print(f"V6: THE LIVING MODEL — ACCUMULATE, DON'T CONSOLIDATE")
    print(f"{'='*120}")
    print(f"{'Method':<22} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<8} {'FF(B)':<8} {'Repairs':<8}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x.get("method","")):
        method = r.get("method","?"); phases = r.get("phases",{}); keys = list(phases.keys())
        if len(keys) < 3: print(f"{method:<22} (incomplete)"); continue
        fp = phases[keys[-1]].get("perplexity",{})
        a, b, c = fp.get("A",0), fp.get("B",0), fp.get("C",0)
        ff_a = a / phases["A"].get("perplexity",{}).get("A",1) if a > 0 else 0
        ff_b = b / phases["B"].get("perplexity",{}).get("B",1) if b > 0 else 0
        rep = phases[keys[-1]].get("repairs",0)
        print(f"{method:<22} {a:<10.1f} {b:<10.1f} {c:<10.1f} {ff_a:<8.2f}x {ff_b:<8.2f}x {rep:<8}")
    print(f"{'='*120}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6: The Living Model")
    parser.add_argument("--method", default=None, choices=METHODS)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args, _ = parser.parse_known_args()
    if args.quick: run_quick()
    elif args.grid: run_grid()
    elif args.method: run_experiment(args.method)
    else:
        print("V6: The Living Model — Accumulate, Don't Consolidate")
        print("run_grid() | run_quick() | run_experiment('accumulate_avr')")
        print()
        run_grid()
