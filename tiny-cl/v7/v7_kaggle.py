"""
V7: Orthogonal Accumulate — Fix the Merge Corruption Problem

V6 DIAGNOSTIC RESULT:
  LoRA interference = 0.00 across all domains
  Forgetting is 100% from base weight corruption during merge
  Adding domain B's delta to base destroys domain A's representations

THE FIX:
  Don't let domain B's LoRA operate in the same subspace as domain A's.
  Two complementary mechanisms:

  1. ORTHOGONALITY PENALTY (during training):
     When training domain B, penalize B's LoRA matrices from overlapping
     with A's stored LoRA matrices. This forces B to learn in a different
     subspace, so B's delta doesn't corrupt A's committed knowledge.

     loss_ortho = ||A_B @ A_A^T||_F^2 + ||B_B^T @ B_A||_F^2

  2. EWC-WEIGHTED MERGE (during commit):
     After training each domain, compute Fisher importance of base weights
     for all previous domains. When merging new delta, scale it down
     where Fisher is high (important for old domains).

     delta_scaled = delta / (1 + lambda * fisher)

METHODS:
  naive          = baseline (all LoRA, no CL) — same as V5/V6
  ortho          = conv LoRA + orthogonality penalty + accumulate
  ortho_ewc      = conv LoRA + orthogonality + EWC-weighted merge
  ewc_only       = conv LoRA + EWC-weighted merge (no orthogonality)

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
    results_dir: str = "v7_results"

@dataclass
class OrthoConfig:
    ortho_lambda: float = 1.0
    ewc_lambda: float = 5000.0
    fisher_n_samples: int = 200
    fisher_batch_size: int = 4

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text"),
}
METHODS = ["naive", "ortho", "ewc_only", "ortho_ewc"]
ORTHO_CFG = OrthoConfig()


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
# MERGE FUNCTIONS
# ──────────────────────────────────────────────

def merge_conv_lora_into_base(model):
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
            nn.init.kaiming_uniform_(module.lora_A[adapter_name].weight, a=math.sqrt(5))
            module.lora_B[adapter_name].weight.data.zero_()
        if adapter_name in module.merged_adapters:
            module.merged_adapters.remove(adapter_name)
        merged_count += 1
    print(f"  [COMMIT] Merged {merged_count} conv LoRA into base (standard)")
    return merged_count


def merge_conv_lora_with_ewc(model, fisher_dict, ewc_lambda):
    from peft.tuners.lora.layer import LoraLayer
    merged_count = 0
    adapter_name = "default"
    total_masked = 0
    total_params = 0

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if adapter_name not in module.lora_A: continue

        with torch.no_grad():
            A = module.lora_A[adapter_name].weight.data
            B = module.lora_B[adapter_name].weight.data
            scaling = module.scaling[adapter_name]
            delta = (B @ A) * scaling

            base_key = f"{name}.base_layer.weight"
            if base_key in fisher_dict:
                fish = fisher_dict[base_key].to(delta.device)
                f_max = fish.max()
                if f_max > 0:
                    f_norm = fish / f_max
                else:
                    f_norm = torch.zeros_like(fish)
                mask = 1.0 / (1.0 + ewc_lambda * f_norm)
                delta = delta * mask
                total_masked += (mask < 0.5).sum().item()
                total_params += mask.numel()

            module.base_layer.weight.data += delta
            nn.init.kaiming_uniform_(module.lora_A[adapter_name].weight, a=math.sqrt(5))
            module.lora_B[adapter_name].weight.data.zero_()

        if adapter_name in module.merged_adapters:
            module.merged_adapters.remove(adapter_name)
        merged_count += 1

    pct = 100 * total_masked / max(total_params, 1)
    print(f"  [COMMIT-EWC] Merged {merged_count} conv LoRA (lambda={ewc_lambda:.0f}, {pct:.1f}% params protected)")
    return merged_count


# ──────────────────────────────────────────────
# FISHER COMPUTATION
# ──────────────────────────────────────────────

def compute_fisher_conv_base(model, dataset, device, n_batches=200, batch_size=4):
    model.train()
    fisher = {}
    for name, param in model.named_parameters():
        is_conv = False
        for idx in CONV_LAYER_IDS:
            if f"layers.{idx}." in name:
                is_conv = True
                break
        if is_conv and "base_layer.weight" in name:
            fisher[name] = torch.zeros_like(param.data).cpu()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n = 0
    for batch in loader:
        if n >= n_batches: break
        model.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        out.loss.backward()
        for name, param in model.named_parameters():
            if name in fisher and param.grad is not None:
                fisher[name] += param.grad.data.cpu().pow(2)
        n += 1
        del out

    for name in fisher:
        fisher[name] /= max(n, 1)

    model.zero_grad()
    total_p = sum(f.numel() for f in fisher.values())
    print(f"  [FISHER] {len(fisher)} conv base weights ({total_p:,} params, ~{total_p*4/1024/1024:.0f}MB)")
    return fisher

def accumulate_fisher(old_fisher, new_fisher):
    if old_fisher is None:
        return new_fisher
    merged = {}
    for name in new_fisher:
        if name in old_fisher:
            merged[name] = old_fisher[name] + new_fisher[name]
        else:
            merged[name] = new_fisher[name]
    return merged


# ──────────────────────────────────────────────
# ORTHOGONALITY
# ──────────────────────────────────────────────

def store_lora_state(model):
    from peft.tuners.lora.layer import LoraLayer
    state = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue
        state[name] = {
            "A": module.lora_A["default"].weight.data.cpu().clone(),
            "B": module.lora_B["default"].weight.data.cpu().clone(),
        }
    return state

def compute_ortho_loss(model, stored_loras, device):
    from peft.tuners.lora.layer import LoraLayer
    ortho_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue

        cur_A = module.lora_A["default"].weight
        cur_B = module.lora_B["default"].weight

        for prev_pk in stored_loras:
            if name not in stored_loras[prev_pk]: continue
            prev_A = stored_loras[prev_pk][name]["A"].to(device)
            prev_B = stored_loras[prev_pk][name]["B"].to(device)

            # A orthogonality: ||cur_A @ prev_A^T||_F^2
            ortho_loss = ortho_loss + torch.norm(cur_A @ prev_A.T, p='fro')**2
            n_terms += 1

            # B orthogonality: ||cur_B^T @ prev_B||_F^2
            ortho_loss = ortho_loss + torch.norm(cur_B.T @ prev_B, p='fro')**2
            n_terms += 1

    if n_terms > 0:
        ortho_loss = ortho_loss / n_terms

    return ortho_loss


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
        self.total_repairs, self.total_verifies = 0, 0
    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True
    def on_phase_end(self, model, pk, dataset, device):
        self.completed.append(pk)
    def on_step_end(self, model, pk, step, device): pass
    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class OrthoMethod:
    """Conv LoRA + orthogonality penalty + standard merge."""
    def __init__(self):
        self.name = "ortho"
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
        self.stored_loras = {}
        self.ortho_lambda = ORTHO_CFG.ortho_lambda

    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        self.stored_loras[pk] = store_lora_state(model)
        n_stored = len(self.stored_loras)
        print(f"  [ORTHO] Stored LoRA for {pk} ({n_stored} total stored domains)")
        merge_conv_lora_into_base(model)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device): pass

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        lm_loss = out.loss
        if self.completed:
            ortho_loss = compute_ortho_loss(model, self.stored_loras, device)
            total_loss = lm_loss + self.ortho_lambda * ortho_loss
            return total_loss, {"lm_loss": lm_loss.item(), "ortho_loss": ortho_loss.item()}
        else:
            return lm_loss, {"lm_loss": lm_loss.item(), "ortho_loss": 0.0}


class EWCOnlyMethod:
    """Conv LoRA + EWC-weighted merge (no orthogonality during training)."""
    def __init__(self):
        self.name = "ewc_only"
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
        self.cumulative_fisher = None
        self.ewc_lambda = ORTHO_CFG.ewc_lambda

    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        domain_fisher = compute_fisher_conv_base(
            model, dataset, device,
            n_batches=ORTHO_CFG.fisher_n_samples,
            batch_size=ORTHO_CFG.fisher_batch_size,
        )
        self.cumulative_fisher = accumulate_fisher(self.cumulative_fisher, domain_fisher)
        if self.completed:
            merge_conv_lora_with_ewc(model, self.cumulative_fisher, self.ewc_lambda)
        else:
            merge_conv_lora_into_base(model)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device): pass

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        return out.loss, {"lm_loss": out.loss.item()}


class OrthoEWCMethod:
    """Conv LoRA + orthogonality penalty + EWC-weighted merge."""
    def __init__(self):
        self.name = "ortho_ewc"
        self.completed, self.extra_steps = [], 0
        self.total_repairs, self.total_verifies = 0, 0
        self.stored_loras = {}
        self.cumulative_fisher = None
        self.ortho_lambda = ORTHO_CFG.ortho_lambda
        self.ewc_lambda = ORTHO_CFG.ewc_lambda

    def on_phase_start(self, model, pk):
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True

    def on_phase_end(self, model, pk, dataset, device):
        self.stored_loras[pk] = store_lora_state(model)
        domain_fisher = compute_fisher_conv_base(
            model, dataset, device,
            n_batches=ORTHO_CFG.fisher_n_samples,
            batch_size=ORTHO_CFG.fisher_batch_size,
        )
        self.cumulative_fisher = accumulate_fisher(self.cumulative_fisher, domain_fisher)
        if self.completed:
            merge_conv_lora_with_ewc(model, self.cumulative_fisher, self.ewc_lambda)
        else:
            merge_conv_lora_into_base(model)
        self.completed.append(pk)

    def on_step_end(self, model, pk, step, device): pass

    def compute_loss(self, model, batch, device):
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        lm_loss = out.loss
        if self.completed:
            ortho_loss = compute_ortho_loss(model, self.stored_loras, device)
            total_loss = lm_loss + self.ortho_lambda * ortho_loss
            return total_loss, {"lm_loss": lm_loss.item(), "ortho_loss": ortho_loss.item()}
        else:
            return lm_loss, {"lm_loss": lm_loss.item(), "ortho_loss": 0.0}


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
    elif method_name == "ortho": method = OrthoMethod()
    elif method_name == "ewc_only": method = EWCOnlyMethod()
    elif method_name == "ortho_ewc": method = OrthoEWCMethod()
    else: raise ValueError(f"Unknown: {method_name}")

    results = {"method": method_name, "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        dataset = phases_data[pk]
        method.on_phase_start(model, pk)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
        gs, tl, ol_total = 0, 0.0, 0.0
        t0 = time.time()
        print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} ({len(dataset)} samples)")

        for epoch in range(tc.epochs_per_phase):
            for batch in loader:
                model.train()
                loss, metrics = method.compute_loss(model, batch, device)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step()
                tl += metrics["lm_loss"]
                ol_total += metrics.get("ortho_loss", 0.0)
                gs += 1
                method.on_step_end(model, pk, gs, device)
                if gs % 100 == 0:
                    ol_str = f" ortho={ol_total/gs:.4f}" if ol_total > 0 else ""
                    print(f"    step {gs} | lm={tl/gs:.4f}{ol_str}")

        method.on_phase_end(model, pk, dataset, device)
        model.train()
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)

        ev = eval_all(model, val_ds, method.completed, device, tc.eval_samples)
        ev["avg_loss"] = tl / max(gs, 1)
        ev["ortho_loss_avg"] = ol_total / max(gs, 1) if ol_total > 0 else 0
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
    os.makedirs("v7_results", exist_ok=True)
    done = [f.replace(".json","") for f in os.listdir("v7_results") if f.endswith(".json") and f != "full_grid.json"]
    if done: print(f"  Already done: {done}")
    for meth in methods:
        if meth in done:
            print(f"\n{'*'*60}\n  SKIP {meth}")
            with open(os.path.join("v7_results", f"{meth}.json")) as f: all_res.append(json.load(f))
            continue
        print(f"\n{'*'*60}")
        try: all_res.append(run_experiment(meth))
        except Exception as e: print(f"  FAILED: {e}"); import traceback; traceback.print_exc()
    with open("v7_results/full_grid.json", "w") as f: json.dump(all_res, f, indent=2)
    print_summary(all_res)
    return all_res

def run_quick():
    print("QUICK: naive + ortho")
    return run_grid(["naive", "ortho"])

def print_summary(results=None):
    if results is None:
        results = []
        for f in os.listdir("v7_results"):
            if f.endswith(".json") and f != "full_grid.json":
                with open(os.path.join("v7_results", f)) as fh: results.append(json.load(fh))
    if not results: print("No results"); return
    print(f"\n{'='*130}")
    print(f"V7: ORTHOGONAL ACCUMULATE — FIX THE MERGE CORRUPTION")
    print(f"{'='*130}")
    print(f"{'Method':<20} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<10} {'FF(B)':<10} {'B/A Ratio':<10} {'Verdict'}")
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
        print(f"{method:<20} {a:<10.1f} {b:<10.1f} {c:<10.1f} {ff_a:<10.2f}x {ff_b:<10.2f}x {ff_a/ff_b:<10.2f}  {verdict}")
    print(f"{'='*130}")
    print(f"\nV6 baseline: FF(A)=1.52x, FF(B)=1.33x")
    print(f"V7 target: FF(A) < 1.2x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V7: Orthogonal Accumulate")
    parser.add_argument("--method", default=None, choices=METHODS)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args, _ = parser.parse_known_args()
    if args.quick: run_quick()
    elif args.grid: run_grid()
    elif args.method: run_experiment(args.method)
    else: run_grid()
