"""
V8.1: SLAO + Norm Stat Fix

V8 O-LoRA result showed FF(A)=1.23x even with frozen LoRA per task.
That shouldn't happen — A's LoRA is exact, base weights unchanged.

Hypothesis: norm layers (operator_norm) have running statistics that
drift during B/C training. These buffers (running_mean, running_var)
affect ALL inputs, including domain A's.

This script tests:
1. slao_fixed    = SLAO with time-aware scaling (LoRA->LoRA merge)
2. slao_frozen_norm = same + freeze ALL norm layer stats during training
3. olora_frozen_norm = O-LoRA + freeze norm stats (upper bound)

If freezing norm stats drops FF(A) from 1.23x to ~1.0x for O-LoRA,
we found the second source of forgetting.

USAGE: Copy-paste into a single Kaggle cell. Run it.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "datasets", "accelerate"])

import os, json, time, random, math, gc
from dataclasses import dataclass, field
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
    results_dir: str = "v8_results"

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text"),
}
METHODS = ["slao_fixed", "slao_frozen_norm", "olora_frozen_norm"]


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
# MODEL + LoRA
# ──────────────────────────────────────────────

def is_conv_module(name):
    for idx in CONV_LAYER_IDS:
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


# ──────────────────────────────────────────────
# NORM LAYER HELPERS
# ──────────────────────────────────────────────

def inspect_norm_layers(model):
    """Find all norm layers and check if they have running statistics."""
    norm_info = []
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ["norm", "ln", "layer_norm"]):
            has_buffers = False
            for buf_name, buf in module.named_buffers():
                has_buffers = True
                norm_info.append({
                    "name": name,
                    "type": type(module).__name__,
                    "buffer": buf_name,
                    "shape": tuple(buf.shape),
                })
            if not has_buffers:
                norm_info.append({
                    "name": name,
                    "type": type(module).__name__,
                    "buffer": None,
                    "shape": None,
                })
    print(f"  [NORM-INSPECT] Found {len(norm_info)} norm layers:")
    for info in norm_info[:10]:
        buf_str = f" buffer={info['buffer']} shape={info['shape']}" if info['buffer'] else " (no buffers)"
        print(f"    {info['name']} ({info['type']}){buf_str}")
    if len(norm_info) > 10:
        print(f"    ... and {len(norm_info)-10} more")
    return norm_info


def save_norm_stats(model):
    """Save all norm layer running statistics (buffers)."""
    stats = {}
    for name, module in model.named_modules():
        buffers = {}
        for buf_name, buf in module.named_buffers():
            buffers[buf_name] = buf.data.cpu().clone()
        if buffers:
            stats[name] = buffers
    n = sum(b.numel() for s in stats.values() for b in s.values())
    print(f"  [NORM-SAVE] Saved stats from {len(stats)} layers ({n:,} values)")
    return stats


def restore_norm_stats(model, stats):
    """Restore norm layer running statistics from saved state."""
    n_restored = 0
    for name, module in model.named_modules():
        if name in stats:
            for buf_name, buf in module.named_buffers():
                if buf_name in stats[name]:
                    buf.data.copy_(stats[name][buf_name].to(buf.device))
                    n_restored += 1
    print(f"  [NORM-RESTORE] Restored {n_restored} buffers")


def freeze_norm_stats(model):
    """Freeze all norm layer running statistics (set eval mode on norms)."""
    n_frozen = 0
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ["norm", "ln", "layer_norm"]):
            module.eval()  # Switches BatchNorm/LayerNorm to eval mode (no stat updates)
            n_frozen += 1
    print(f"  [NORM-FREEZE] Froze {n_frozen} norm layers (eval mode)")
    return n_frozen


def unfreeze_norm_stats(model):
    """Unfreeze norm layers (set train mode)."""
    n = 0
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ["norm", "ln", "layer_norm"]):
            module.train()
            n += 1
    print(f"  [NORM-UNFREEZE] Unfroze {n} norm layers")


# ──────────────────────────────────────────────
# ORTHOGONAL INIT
# ──────────────────────────────────────────────

def extract_orthogonal_basis(model):
    from peft.tuners.lora.layer import LoraLayer
    basis = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()
        B = module.lora_B["default"].weight.data.float()
        _, _, V_A = torch.linalg.svd(A, full_matrices=False)
        U_B, _, _ = torch.linalg.svd(B, full_matrices=False)
        basis[name] = {"A_right": V_A, "B_left": U_B}
    print(f"  [ORTHO-BASIS] Extracted from {len(basis)} conv LoRA layers")
    return basis


def initialize_orthogonal_lora(model, prev_basis):
    from peft.tuners.lora.layer import LoraLayer
    n_init = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue
        if name not in prev_basis: continue
        A = module.lora_A["default"].weight.data
        B = module.lora_B["default"].weight.data
        V = prev_basis[name]["A_right"].to(A.device)
        U = prev_basis[name]["B_left"].to(B.device)
        proj_A = V.T @ V
        null_A = torch.eye(proj_A.shape[0], device=A.device) - proj_A
        A_orth = A @ null_A
        proj_B = U @ U.T
        null_B = torch.eye(proj_B.shape[0], device=B.device) - proj_B
        B_orth = null_B @ B
        if A_orth.norm() > 1e-8:
            A_orth = A_orth * (A.norm() / A_orth.norm())
        if B_orth.norm() > 1e-8:
            B_orth = B_orth * (B.norm() / B_orth.norm())
        module.lora_A["default"].weight.data.copy_(A_orth)
        module.lora_B["default"].weight.data.copy_(B_orth)
        n_init += 1
    print(f"  [ORTHO-INIT] Projected {n_init} LoRA layers onto orthogonal subspace")


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
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")

def eval_all(model, val_ds, phases, device, max_samp=1024):
    return {"perplexity": {pk: compute_ppl(model, val_ds[pk], device, max_samp) for pk in phases if pk in val_ds}}


# ──────────────────────────────────────────────
# METHODS
# ──────────────────────────────────────────────

class SLAOFixedMethod:
    """SLAO with time-aware scaling, no norm freezing."""
    def __init__(self):
        self.name = "slao_fixed"
        self.completed, self.extra_steps = [], 0
        self.prev_basis = None
        self.task_num = 0
        self.accumulated_lora = None

    def on_phase_start(self, model, pk):
        self.task_num += 1
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True
        if self.prev_basis is not None:
            for n, p in model.named_parameters():
                if "lora_A" in n and is_conv_module(n): nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if "lora_B" in n and is_conv_module(n): p.data.zero_()
            initialize_orthogonal_lora(model, self.prev_basis)

    def on_phase_end(self, model, pk, dataset, device):
        if self.accumulated_lora is not None and self.task_num > 1:
            alpha = (self.task_num - 1) / self.task_num
            for n, p in model.named_parameters():
                if n in self.accumulated_lora and "lora_" in n and is_conv_module(n):
                    old_val = self.accumulated_lora[n].to(device)
                    p.data.copy_(alpha * old_val + (1 - alpha) * p.data)
            print(f"  [SLAO] Task {self.task_num}: merged (alpha={alpha:.3f})")
        else:
            print(f"  [SLAO] Task {self.task_num}: first task")
        self.accumulated_lora = {
            n: p.data.cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n and is_conv_module(n)
        }
        self.prev_basis = extract_orthogonal_basis(model)
        self.completed.append(pk)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class SLAOFrozenNormMethod:
    """SLAO + freeze norm layer stats during training."""
    def __init__(self):
        self.name = "slao_frozen_norm"
        self.completed, self.extra_steps = [], 0
        self.prev_basis = None
        self.task_num = 0
        self.accumulated_lora = None
        self.initial_norm_stats = None

    def on_phase_start(self, model, pk):
        self.task_num += 1

        # On first task, save initial norm stats
        if self.initial_norm_stats is None:
            self.initial_norm_stats = save_norm_stats(model)

        # Freeze norm layers (eval mode = no running stat updates)
        freeze_norm_stats(model)

        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True
        if self.prev_basis is not None:
            for n, p in model.named_parameters():
                if "lora_A" in n and is_conv_module(n): nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if "lora_B" in n and is_conv_module(n): p.data.zero_()
            initialize_orthogonal_lora(model, self.prev_basis)

    def on_phase_end(self, model, pk, dataset, device):
        if self.accumulated_lora is not None and self.task_num > 1:
            alpha = (self.task_num - 1) / self.task_num
            for n, p in model.named_parameters():
                if n in self.accumulated_lora and "lora_" in n and is_conv_module(n):
                    old_val = self.accumulated_lora[n].to(device)
                    p.data.copy_(alpha * old_val + (1 - alpha) * p.data)
            print(f"  [SLAO-FN] Task {self.task_num}: merged (alpha={alpha:.3f})")
        else:
            print(f"  [SLAO-FN] Task {self.task_num}: first task")

        self.accumulated_lora = {
            n: p.data.cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n and is_conv_module(n)
        }
        self.prev_basis = extract_orthogonal_basis(model)
        self.completed.append(pk)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class OLoRAFrozenNormMethod:
    """O-LoRA + freeze norm stats. Upper bound with norm protection."""
    def __init__(self):
        self.name = "olora_frozen_norm"
        self.completed, self.extra_steps = [], 0
        self.prev_basis = None
        self.frozen_lora_states = {}
        self.initial_norm_stats = None

    def on_phase_start(self, model, pk):
        if self.initial_norm_stats is None:
            self.initial_norm_stats = save_norm_stats(model)
        freeze_norm_stats(model)

        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True
        if self.prev_basis is not None:
            for n, p in model.named_parameters():
                if "lora_A" in n and is_conv_module(n): nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if "lora_B" in n and is_conv_module(n): p.data.zero_()
            initialize_orthogonal_lora(model, self.prev_basis)

    def on_phase_end(self, model, pk, dataset, device):
        state = {}
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n):
                state[n] = p.data.cpu().clone()
        self.frozen_lora_states[pk] = state
        n_p = sum(v.numel() for v in state.values())
        print(f"  [O-LORA-FN] Froze LoRA for {pk} ({n_p:,} params)")
        self.prev_basis = extract_orthogonal_basis(model)
        self.completed.append(pk)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}

    def load_task_lora(self, model, pk, device):
        if pk not in self.frozen_lora_states: return
        state = self.frozen_lora_states[pk]
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n].to(device))


# ──────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────

def run_experiment(method_name, seed=42):
    torch.manual_seed(seed); random.seed(seed)
    tc = TrainConfig(); mcfg = MODEL_CONFIG; device = tc.device

    print(f"\n{'#'*60}")
    print(f"# {method_name} | device={device}")
    print(f"{'#'*60}")

    model, tokenizer = create_model_conv_only(mcfg, device)

    # Inspect norm layers on first run
    inspect_norm_layers(model)

    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    if method_name == "slao_fixed": method = SLAOFixedMethod()
    elif method_name == "slao_frozen_norm": method = SLAOFrozenNormMethod()
    elif method_name == "olora_frozen_norm": method = OLoRAFrozenNormMethod()
    else: raise ValueError(f"Unknown: {method_name}")

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
                # Re-freeze norm layers (model.train() may have unfrozen them)
                if "frozen_norm" in method_name:
                    freeze_norm_stats(model)
                loss, metrics = method.compute_loss(model, batch, device)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += metrics["lm_loss"]; gs += 1
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

        method.on_phase_end(model, pk, dataset, device)

        # For olora: load task-specific LoRA; restore norm stats for eval
        if "olora" in method_name and hasattr(method, 'load_task_lora'):
            ppls = {}
            for prev_pk in method.completed:
                method.load_task_lora(model, prev_pk, device)
                # Restore initial norm stats for fair eval
                if method.initial_norm_stats:
                    restore_norm_stats(model, method.initial_norm_stats)
                ppls[prev_pk] = compute_ppl(model, val_ds[prev_pk], device, tc.eval_samples)
            ev = {"perplexity": ppls}
        else:
            ev = eval_all(model, val_ds, method.completed, device, tc.eval_samples)

        ev["avg_loss"] = tl / max(gs, 1)
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
# GRID + SUMMARY
# ──────────────────────────────────────────────

def run_grid(methods=None):
    methods = methods or METHODS
    all_res = []
    os.makedirs("v8_results", exist_ok=True)
    done = [f.replace(".json","") for f in os.listdir("v8_results") if f.endswith(".json") and f != "full_grid.json"]
    if done: print(f"  Already done: {done}")
    for meth in methods:
        if meth in done:
            print(f"\n{'*'*60}\n  SKIP {meth}")
            with open(os.path.join("v8_results", f"{meth}.json")) as f: all_res.append(json.load(f))
            continue
        print(f"\n{'*'*60}")
        try: all_res.append(run_experiment(meth))
        except Exception as e: print(f"  FAILED: {e}"); import traceback; traceback.print_exc()
    with open("v8_results/full_grid_v81.json", "w") as f: json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res

def print_summary(results=None):
    if results is None:
        results = []
        for f in os.listdir("v8_results"):
            if f.endswith(".json") and f not in ("full_grid.json", "full_grid_v81.json"):
                with open(os.path.join("v8_results", f)) as fh: results.append(json.load(fh))
    if not results: print("No results"); return
    print(f"\n{'='*140}")
    print(f"V8.1: SLAO + NORM STAT FREEZE")
    print(f"{'='*140}")
    print(f"{'Method':<25} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<10} {'FF(B)':<10} {'Verdict'}")
    print("-" * 140)
    for r in sorted(results, key=lambda x: x.get("method","")):
        method = r.get("method","?"); phases = r.get("phases",{}); keys = list(phases.keys())
        if len(keys) < 3: print(f"{method:<25} (incomplete)"); continue
        fp = phases[keys[-1]].get("perplexity",{})
        a, b, c = fp.get("A",0), fp.get("B",0), fp.get("C",0)
        ff_a = a / phases["A"].get("perplexity",{}).get("A",1) if a > 0 else 0
        ff_b = b / phases["B"].get("perplexity",{}).get("B",1) if b > 0 else 0
        if ff_a < 1.05: verdict = "NEAR PERFECT"
        elif ff_a < 1.1: verdict = "EXCELLENT"
        elif ff_a < 1.3: verdict = "GOOD"
        elif ff_a < 1.5: verdict = "OK"
        else: verdict = "POOR"
        print(f"{method:<25} {a:<10.1f} {b:<10.1f} {c:<10.1f} {ff_a:<10.2f}x {ff_b:<10.2f}x {verdict}")
    print(f"{'='*140}")
    print(f"\nV8 baselines (no norm freeze):")
    print(f"  slao_simple: FF(A)=1.26x, FF(B)=1.24x")
    print(f"  olora:       FF(A)=1.23x, FF(B)=1.06x")
    print(f"\nIf frozen norm improves olora FF(A) from 1.23x -> ~1.0x,")
    print(f"then norm stat drift is the second source of forgetting.")


if __name__ == "__main__":
    run_grid()
