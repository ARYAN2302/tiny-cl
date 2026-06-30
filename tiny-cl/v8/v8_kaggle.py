"""
V8: Merge Before Forget — Based on ICLR 2026 Paper

Paper: "Merge before Forget: A Single LoRA Continual Learning via
        Continual Merging" (Qiao & Mahdavi, ICLR 2026)

Key insight from V6 diagnostic: Merging LoRA into BASE weights causes
corruption (FF=1.52x). Nobody in the literature merges into base.

This paper's solution (SLAO):
  1. Keep a SINGLE LoRA that accumulates all task knowledge
  2. Before training new task, extract orthogonal basis from current LoRA
     and use it to initialize the new task's LoRA (O-LoRA style)
  3. After training, the LoRA already has both old+new knowledge
     (no merge needed — it's the same LoRA!)
  4. Time-aware scaling: for separate-LoRA variants, weight old vs new

The key: NEVER touch base weights. Stay in low-rank LoRA space.

METHODS:
  naive       = baseline (all LoRA, no CL)
  slao        = SLAO: orthogonal init, single accumulating LoRA
  slao_simple = same but without time-aware scaling
  olora       = O-LoRA baseline: orthogonal init, keep separate LoRAs

USAGE: Copy-paste this entire file into a single Kaggle cell. Run it.
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
METHODS = ["naive", "slao", "slao_simple", "olora"]


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

def create_model_all_lora(mcfg, device):
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(mcfg, device)
    all_targets = list(set(mcfg.conv_targets + ["q_proj", "v_proj"]))
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=all_targets, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ──────────────────────────────────────────────
# SLAO CORE: ORTHOGONAL INIT
# ──────────────────────────────────────────────

def extract_orthogonal_basis(model):
    """Extract the subspaces occupied by the current LoRA.
    
    A: [rank, in_features] — rows span the input subspace
    B: [out_features, rank] — columns span the output subspace
    
    We use SVD to find these subspaces for orthogonal initialization.
    """
    from peft.tuners.lora.layer import LoraLayer
    basis = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue

        A = module.lora_A["default"].weight.data.float()
        B = module.lora_B["default"].weight.data.float()

        # SVD of A: right singular vectors span the input subspace
        _, _, V_A = torch.linalg.svd(A, full_matrices=False)
        # SVD of B: left singular vectors span the output subspace
        U_B, _, _ = torch.linalg.svd(B, full_matrices=False)

        basis[name] = {"A_right": V_A, "B_left": U_B}

    print(f"  [ORTHO-BASIS] Extracted from {len(basis)} conv LoRA layers")
    return basis


def initialize_orthogonal_lora(model, prev_basis):
    """Project new LoRA initialization onto the null space of previous LoRA.
    
    This ensures the new task's LoRA operates in a subspace that doesn't
    interfere with previous tasks.
    """
    from peft.tuners.lora.layer import LoraLayer
    n_init = 0

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue
        if name not in prev_basis: continue

        A = module.lora_A["default"].weight.data  # [rank, in_features]
        B = module.lora_B["default"].weight.data  # [out_features, rank]

        V = prev_basis[name]["A_right"].to(A.device)   # [rank, in_features]
        U = prev_basis[name]["B_left"].to(B.device)     # [out_features, rank]

        # Project A onto null space of V's row space
        # null_proj = I - V^T @ V
        proj_A = V.T @ V
        null_A = torch.eye(proj_A.shape[0], device=A.device) - proj_A
        A_orth = A @ null_A

        # Project B onto null space of U's column space
        # null_proj = I - U @ U^T
        proj_B = U @ U.T
        null_B = torch.eye(proj_B.shape[0], device=B.device) - proj_B
        B_orth = null_B @ B

        # Re-normalize to maintain scale
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
    model.train()
    return math.exp(tot_loss / tot_tok) if tot_tok > 0 else float("inf")

def eval_all(model, val_ds, phases, device, max_samp=1024):
    return {"perplexity": {pk: compute_ppl(model, val_ds[pk], device, max_samp) for pk in phases if pk in val_ds}}


# ──────────────────────────────────────────────
# METHODS
# ──────────────────────────────────────────────

class NaiveMethod:
    def __init__(self):
        self.name = "naive"
        self.completed, self.extra_steps = [], 0
    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True
    def on_phase_end(self, model, pk, dataset, device):
        self.completed.append(pk)
    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class SLAOMethod:
    """SLAO: Single LoRA with Orthogonal init — from ICLR 2026 paper.
    
    CORRECT implementation: The paper trains SEPARATE LoRAs per task (with
    ortho init), then merges them into a single LoRA using time-aware scaling.
    
    Key: the merged LoRA stays in LoRA space (never touches base weights).
    
    Algorithm:
    1. Task t: Train a new LoRA orthogonally initialized to previous tasks
    2. After training, merge with accumulated LoRA:
       A_acc = alpha * A_acc + (1-alpha) * A_new
       B_acc = alpha * B_acc + (1-alpha) * B_new
       where alpha = (t-1)/t
    3. The accumulated LoRA captures all tasks 1..t
    """
    def __init__(self, use_time_scaling=True):
        self.name = "slao" if use_time_scaling else "slao_simple"
        self.completed, self.extra_steps = [], 0
        self.prev_basis = None
        self.task_num = 0
        self.use_time_scaling = use_time_scaling
        # Accumulated LoRA: the merged result of all previous tasks
        self.accumulated_lora = None  # Dict of param_name -> tensor

    def on_phase_start(self, model, pk):
        self.task_num += 1
        # FIX: enable ALL LoRA params (not just conv) — out_proj exists in attn layers too
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if self.prev_basis is not None:
            # STEP 1: Re-init LoRA fresh (this is the NEW task's LoRA)
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if "lora_B" in n:
                    p.data.zero_()

            # STEP 2: Project onto orthogonal subspace of accumulated LoRA
            initialize_orthogonal_lora(model, self.prev_basis)
        # First task: train normally, no special init

    def on_phase_end(self, model, pk, dataset, device):
        # STEP 3: Merge new task's LoRA with accumulated LoRA
        if self.accumulated_lora is not None and self.task_num > 1:
            # Time-aware scaling: alpha = (t-1)/t
            if self.use_time_scaling:
                alpha = (self.task_num - 1) / self.task_num
            else:
                alpha = 0.5  # Simple average

            for n, p in model.named_parameters():
                if n in self.accumulated_lora and "lora_" in n:
                    old_val = self.accumulated_lora[n].to(device)
                    # Weighted merge: alpha * old + (1-alpha) * new
                    p.data.copy_(alpha * old_val + (1 - alpha) * p.data)

            print(f"  [SLAO] Task {self.task_num}: merged with accumulated LoRA (alpha={alpha:.3f})")
        else:
            print(f"  [SLAO] Task {self.task_num}: first task, saved as accumulated LoRA")

        # Save current (merged) LoRA as accumulated for next task
        # FIX: save ALL LoRA params, not just conv-layer ones
        self.accumulated_lora = {
            n: p.data.cpu().clone() for n, p in model.named_parameters()
            if "lora_" in n
        }
        n_p = sum(v.numel() for v in self.accumulated_lora.values())
        print(f"    Accumulated LoRA: {n_p:,} params (~{n_p*4/1024/1024:.1f}MB)")

        # Extract basis for next task's orthogonal init
        self.prev_basis = extract_orthogonal_basis(model)
        self.completed.append(pk)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class OLoRAMethod:
    """O-LoRA baseline: orthogonal init, keep SEPARATE frozen LoRAs per task.
    
    Upper bound: each task gets its own LoRA, no interference possible.
    At eval, swap in the correct task's LoRA.
    """
    def __init__(self):
        self.name = "olora"
        self.completed, self.extra_steps = [], 0
        self.prev_basis = None
        self.frozen_lora_states = {}

    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if self.prev_basis is not None:
            for n, p in model.named_parameters():
                if "lora_A" in n:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if "lora_B" in n:
                    p.data.zero_()
            initialize_orthogonal_lora(model, self.prev_basis)

    def on_phase_end(self, model, pk, dataset, device):
        # Freeze current LoRA by saving state — ALL LoRA params (conv + attn)
        state = {}
        for n, p in model.named_parameters():
            if "lora_" in n:
                state[n] = p.data.cpu().clone()
        self.frozen_lora_states[pk] = state
        n_p = sum(v.numel() for v in state.values())
        print(f"  [O-LORA] Froze LoRA for {pk} ({n_p:,} params, ~{n_p*4/1024/1024:.1f}MB)")

        self.prev_basis = extract_orthogonal_basis(model)
        self.completed.append(pk)

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}

    def load_task_lora(self, model, pk, device):
        """Load a specific task's frozen LoRA for evaluation."""
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

    if method_name == "naive":
        model, tokenizer = create_model_all_lora(mcfg, device)
    else:
        model, tokenizer = create_model_conv_only(mcfg, device)

    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    if method_name == "naive": method = NaiveMethod()
    elif method_name == "slao": method = SLAOMethod(use_time_scaling=True)
    elif method_name == "slao_simple": method = SLAOMethod(use_time_scaling=False)
    elif method_name == "olora": method = OLoRAMethod()
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
                loss, metrics = method.compute_loss(model, batch, device)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += metrics["lm_loss"]; gs += 1
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

        method.on_phase_end(model, pk, dataset, device)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)

        # Eval
        if method_name == "olora" and hasattr(method, 'load_task_lora'):
            # O-LoRA: eval each domain with its own LoRA
            ppls = {}
            for prev_pk in method.completed:
                method.load_task_lora(model, prev_pk, device)
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
    with open("v8_results/full_grid.json", "w") as f: json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res

def run_quick():
    print("QUICK: naive + slao + olora")
    return run_grid(["naive", "slao", "olora"])

def print_summary(results=None):
    if results is None:
        results = []
        for f in os.listdir("v8_results"):
            if f.endswith(".json") and f != "full_grid.json":
                with open(os.path.join("v8_results", f)) as fh: results.append(json.load(fh))
    if not results: print("No results"); return
    print(f"\n{'='*130}")
    print(f"V8: MERGE BEFORE FORGET (SLAO — ICLR 2026)")
    print(f"Key: LoRA->LoRA merge, NOT LoRA->base merge")
    print(f"{'='*130}")
    print(f"{'Method':<20} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<10} {'FF(B)':<10} {'Verdict'}")
    print("-" * 130)
    for r in sorted(results, key=lambda x: x.get("method","")):
        method = r.get("method","?"); phases = r.get("phases",{}); keys = list(phases.keys())
        if len(keys) < 3: print(f"{method:<20} (incomplete)"); continue
        fp = phases[keys[-1]].get("perplexity",{})
        a, b, c = fp.get("A",0), fp.get("B",0), fp.get("C",0)
        ff_a = a / phases["A"].get("perplexity",{}).get("A",1) if a > 0 else 0
        ff_b = b / phases["B"].get("perplexity",{}).get("B",1) if b > 0 else 0
        if ff_a < 1.1: verdict = "EXCELLENT"
        elif ff_a < 1.3: verdict = "GOOD"
        elif ff_a < 1.5: verdict = "OK"
        else: verdict = "POOR"
        print(f"{method:<20} {a:<10.1f} {b:<10.1f} {c:<10.1f} {ff_a:<10.2f}x {ff_b:<10.2f}x {verdict}")
    print(f"{'='*130}")
    print(f"\nV6 baseline: FF(A)=1.52x (LoRA->base merge = corruption)")
    print(f"V8 target:  FF(A) < 1.2x (LoRA->LoRA merge, no base corruption)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V8: Merge Before Forget")
    parser.add_argument("--method", default=None, choices=METHODS)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args, _ = parser.parse_known_args()
    if args.quick: run_quick()
    elif args.grid: run_grid()
    elif args.method: run_experiment(args.method)
    else: run_grid()
