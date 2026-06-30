"""
V11: SLAO + AVR Combination — Controlled Experiment

Addresses the design tension:
  - AVR repair target = previous merged_state snapshot (constant memory)
  - Repair fires AFTER SLAO merge (clean causal story)
  - Track repair fire count (zero is a real, reportable result)

Methods:
  1. naive     — sequential fine-tuning, no CL
  2. ewc       — EWC regularization (finally)
  3. slao      — SLAO Algorithm 1 (ICLR 2026, arXiv 2512.23017)
  4. slao_avr  — SLAO + AVR verify-repair after merge
  5. naive_ext — naive + extra steps matching slao_avr's repair budget

Checks:
  - Multi-seed (42, 123, 456)
  - Forward (A→B→C) and reverse (C→B→A) order
  - Plasticity cost (newest domain PPL)
  - Repair fire count per seed
  - Module verification
  - Compute-matched dummy baseline

USAGE: Copy-paste into one Kaggle cell. Run it.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "datasets", "accelerate"])

import os, json, time, random, math, gc, copy
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]
FORWARD_ORDER = ["A", "B", "C"]
REVERSE_ORDER = ["C", "B", "A"]

@dataclass
class ModelConfig:
    hf_id: str = "LiquidAI/LFM2.5-350M"
    context_length: int = 512
    batch_size: int = 8
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["in_proj", "out_proj"])

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: list = field(default_factory=lambda: [42, 123, 456])
    eval_samples: int = 1024
    results_dir: str = "v11_results"
    # EWC
    ewc_lambda: float = 5000.0
    fisher_samples: int = 512
    # AVR
    drift_threshold: float = 1.15   # fire repair if PPL > 1.15x best
    repair_alpha: float = 0.1       # pull strength per repair step
    max_repair_steps: int = 10

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text"),
}


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
# MODEL
# ──────────────────────────────────────────────

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

def create_model(mcfg, device):
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(mcfg, device)
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=mcfg.target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Verify what LoRA wraps
    from peft.tuners.lora.layer import LoraLayer
    conv_c, attn_c, other_c = 0, 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        is_conv = any(f"layers.{idx}." in name for idx in CONV_LAYER_IDS)
        is_attn = any(f"layers.{idx}." in name for idx in ATTN_LAYER_IDS)
        if is_conv: conv_c += 1
        elif is_attn: attn_c += 1
        else: other_c += 1
    print(f"  [VERIFY] LoRA: {conv_c} conv + {attn_c} attn + {other_c} other = {conv_c+attn_c+other_c} total")
    return model, tokenizer


# ──────────────────────────────────────────────
# SHARED UTILS
# ──────────────────────────────────────────────

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state, device):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device))

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

def train_phase(model, dataset, mcfg, tc, device, loss_fn=None):
    """Standard training loop for one phase. Returns (steps, total_loss)."""
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
    loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    for epoch in range(tc.epochs_per_phase):
        for batch in loader:
            model.train()
            if loss_fn:
                loss = loss_fn(model, batch, device)
            else:
                out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                loss = out.loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            opt.step(); tl += loss.item(); gs += 1
            if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")
    return gs, tl

def eval_all_domains(model, val_ds, domain_order, device, eval_samples):
    """Evaluate on all domains, return dict {pk: ppl}."""
    ppls = {}
    for pk in domain_order:
        if val_ds.get(pk) is not None:
            ppls[pk] = compute_ppl(model, val_ds[pk], device, eval_samples)
    return ppls


# ──────────────────────────────────────────────
# SLAO CORE (shared between SLAO and SLAO+AVR)
# ──────────────────────────────────────────────

def extract_orthogonal_A(model):
    from peft.tuners.lora.layer import LoraLayer
    ortho_A = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()
        A_T = A.T.contiguous()
        Q, R = torch.linalg.qr(A_T)
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        ortho_A[name] = Q.T
    print(f"  [ORTHO-A] {len(ortho_A)} layers")
    return ortho_A

def initialize_slao(model, ortho_A, prev_ft_B, device):
    from peft.tuners.lora.layer import LoraLayer
    n_init = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        if name in ortho_A:
            module.lora_A["default"].weight.data.copy_(
                ortho_A[name].to(device).to(module.lora_A["default"].weight.data.dtype))
        B_key = f"{name}.lora_B.default.weight"
        if B_key in prev_ft_B:
            module.lora_B["default"].weight.data.copy_(
                prev_ft_B[B_key].to(device).to(module.lora_B["default"].weight.data.dtype))
        n_init += 1
    print(f"  [SLAO-INIT] {n_init} layers: A=ortho, B=prev_ft")

def slao_merge_B(merged_state, ft_state, task_num, device):
    lam = 1.0 / math.sqrt(task_num)
    new_merged = {}
    for key in ft_state:
        ft_val = ft_state[key].to(device)
        if key in merged_state:
            if "lora_A" in key:
                new_merged[key] = ft_val.cpu().clone()
            elif "lora_B" in key:
                old_val = merged_state[key].to(device)
                new_merged[key] = (old_val + lam * (ft_val - old_val)).cpu().clone()
            else:
                new_merged[key] = ft_val.cpu().clone()
        else:
            new_merged[key] = ft_val.cpu().clone()
    print(f"  [SLAO-MERGE] Task {task_num}: A=replace, B=interpolate(λ={lam:.4f})")
    return new_merged


# ──────────────────────────────────────────────
# EWC CORE
# ──────────────────────────────────────────────

def compute_fisher(model, dataset, device, n_samples=512):
    """Compute diagonal Fisher Information Matrix."""
    model.eval()
    fisher = {}
    for n, p in model.named_parameters():
        if "lora_" in n and p.requires_grad:
            fisher[n] = torch.zeros_like(p.data).cpu()

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    n_batches = 0
    for batch in loader:
        if n_batches * 8 >= n_samples: break
        model.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        out.loss.backward()
        for n, p in model.named_parameters():
            if n in fisher and p.grad is not None:
                fisher[n] += p.grad.data.pow(2).cpu()
        n_batches += 1

    for n in fisher:
        fisher[n] /= max(n_batches, 1)

    model.train()
    print(f"  [EWC] Fisher computed from {n_batches} batches")
    return fisher


# ──────────────────────────────────────────────
# AVR CORE
# ──────────────────────────────────────────────

def verify_drift(ppls, best_ppls, completed_phases, threshold):
    """Check if any previous domain has drifted above threshold."""
    drifted = {}
    for pk in completed_phases:
        if pk not in ppls or pk not in best_ppls: continue
        ratio = ppls[pk] / best_ppls[pk] if best_ppls[pk] > 0 else 1.0
        if ratio > threshold:
            drifted[pk] = {"current_ppl": ppls[pk], "best_ppl": best_ppls[pk], "ratio": ratio}
    return drifted

def repair_toward_snapshot(model, snapshot_state, device, alpha):
    """Pull LoRA weights toward snapshot: θ = (1-α)θ + α·θ_snapshot"""
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(device)
            p.data.copy_((1 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj


# ══════════════════════════════════════════════
# METHOD 1: NAIVE
# ══════════════════════════════════════════════

def run_naive(mcfg, tc, phases_data, val_ds, domain_order, device, seed, extra_steps=0):
    """Sequential fine-tuning with no CL. Optionally add extra training steps."""
    tag = f"naive_ext+{extra_steps}" if extra_steps > 0 else "naive"
    print(f"\n{'#'*70}")
    print(f"# {tag.upper()} | seed={seed} | order={'→'.join(domain_order)}")
    print(f"{'#'*70}")

    model, _ = create_model(mcfg, device)
    results = {"method": tag, "seed": seed, "domain_order": domain_order, "phases": {}}
    total_steps = 0

    for task_num, pk in enumerate(domain_order, 1):
        dataset = phases_data[pk]
        gs, tl = train_phase(model, dataset, mcfg, tc, device)
        total_steps += gs

        # Extra steps on current domain (for compute-matched baseline)
        if extra_steps > 0 and task_num > 1:
            print(f"  [COMPUTE-MATCH] Adding {extra_steps} extra steps on {DOMAINS[pk].display_name}")
            for n, p in model.named_parameters():
                if "lora_" in n: p.requires_grad = True
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
            loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
            es, el = 0, 0.0
            for batch in loader:
                model.train()
                out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); el += out.loss.item(); es += 1
                if es >= extra_steps: break
            total_steps += es
            print(f"    {es} extra steps | avg_loss={el/max(es,1):.4f}")

        ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1),
                                  "total_steps": total_steps}
        print(f"  Eval:")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    results["total_steps"] = total_steps
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ══════════════════════════════════════════════
# METHOD 2: EWC
# ══════════════════════════════════════════════

def run_ewc(mcfg, tc, phases_data, val_ds, domain_order, device, seed):
    """Sequential fine-tuning with EWC regularization."""
    print(f"\n{'#'*70}")
    print(f"# EWC (λ={tc.ewc_lambda}) | seed={seed} | order={'→'.join(domain_order)}")
    print(f"{'#'*70}")

    model, _ = create_model(mcfg, device)
    results = {"method": "ewc", "seed": seed, "domain_order": domain_order, "phases": {}}

    fisher_list = []
    star_list = []

    for task_num, pk in enumerate(domain_order, 1):
        dataset = phases_data[pk]

        # Build loss function with EWC penalty
        if fisher_list:
            def make_ewc_loss(fishers, stars, lam):
                def loss_fn(model, batch, device):
                    out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                    ewc_pen = 0.0
                    for fisher, star in zip(fishers, stars):
                        for n, p in model.named_parameters():
                            if n in fisher and "lora_" in n:
                                ewc_pen += (fisher[n].to(device) * (p - star[n].to(device)).pow(2)).sum()
                    return out.loss + lam * ewc_pen
                return loss_fn
            loss_fn = make_ewc_loss(fisher_list, star_list, tc.ewc_lambda)
            print(f"  [EWC] Penalizing with {len(fisher_list)} previous task(s)")
        else:
            loss_fn = None

        gs, tl = train_phase(model, dataset, mcfg, tc, device, loss_fn=loss_fn)

        # Compute Fisher on this task's data after training
        fisher = compute_fisher(model, dataset, device, tc.fisher_samples)
        star_params = get_lora_state(model)
        fisher_list.append(fisher)
        star_list.append(star_params)

        ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval:")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ══════════════════════════════════════════════
# METHOD 3: SLAO
# ══════════════════════════════════════════════

def run_slao(mcfg, tc, phases_data, val_ds, domain_order, device, seed):
    """SLAO: Single LoRA with orthogonal init and B interpolation."""
    print(f"\n{'#'*70}")
    print(f"# SLAO | seed={seed} | order={'→'.join(domain_order)}")
    print(f"{'#'*70}")

    model, _ = create_model(mcfg, device)
    merged_state = None
    prev_ft_state = None

    results = {"method": "slao", "seed": seed, "domain_order": domain_order, "phases": {}}

    for task_num, pk in enumerate(domain_order, 1):
        dataset = phases_data[pk]

        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if task_num == 1:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — standard fine-tune")
        else:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — SLAO init")
            ortho_A = extract_orthogonal_A(model)
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            initialize_slao(model, ortho_A, prev_ft_B, device)

        gs, tl = train_phase(model, dataset, mcfg, tc, device)
        prev_ft_state = get_lora_state(model)

        if merged_state is None:
            merged_state = prev_ft_state.copy()
            print(f"  Task {task_num}: merged = finetuned")
        else:
            merged_state = slao_merge_B(merged_state, prev_ft_state, task_num, device)

        set_lora_state(model, merged_state, device)
        ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval (merged):")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ══════════════════════════════════════════════
# METHOD 4: SLAO + AVR
# ══════════════════════════════════════════════

def run_slao_avr(mcfg, tc, phases_data, val_ds, domain_order, device, seed):
    """SLAO + AVR verify-repair after merge.

    Design:
    - Repair target = previous merged_state snapshot (constant memory)
    - Repair fires AFTER SLAO merge (clean causal story)
    - Track repair fire count — zero is a reportable result
    """
    print(f"\n{'#'*70}")
    print(f"# SLAO+AVR | seed={seed} | order={'→'.join(domain_order)}")
    print(f"# drift_threshold={tc.drift_threshold} | repair_alpha={tc.repair_alpha}")
    print(f"{'#'*70}")

    model, _ = create_model(mcfg, device)
    merged_state = None
    prev_ft_state = None
    merged_snapshot = None   # previous merged_state for repair target
    best_ppls = {}           # best PPL seen for each domain
    completed_phases = []    # phases that have been trained
    total_repair_steps = 0

    results = {"method": "slao_avr", "seed": seed, "domain_order": domain_order,
               "phases": {}, "repair_log": [], "total_repair_steps": 0}

    for task_num, pk in enumerate(domain_order, 1):
        dataset = phases_data[pk]

        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if task_num == 1:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — standard fine-tune")
        else:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — SLAO init")
            ortho_A = extract_orthogonal_A(model)
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            initialize_slao(model, ortho_A, prev_ft_B, device)

        gs, tl = train_phase(model, dataset, mcfg, tc, device)
        prev_ft_state = get_lora_state(model)

        # SLAO merge
        if merged_state is None:
            merged_state = prev_ft_state.copy()
            print(f"  Task {task_num}: merged = finetuned")
        else:
            merged_state = slao_merge_B(merged_state, prev_ft_state, task_num, device)

        set_lora_state(model, merged_state, device)

        # Update best PPLs for current task (right after training)
        post_train_ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)
        if pk not in best_ppls:
            best_ppls[pk] = post_train_ppls[pk]
        completed_phases.append(pk)

        # ── AVR VERIFY + REPAIR ──
        phase_repair_steps = 0
        phase_repair_log = []

        if task_num > 1 and merged_snapshot is not None:
            # Verify: check drift on ALL previous domains
            drifted = verify_drift(post_train_ppls, best_ppls, completed_phases[:-1], tc.drift_threshold)

            if drifted:
                print(f"  [AVR] DRIFT DETECTED on {list(drifted.keys())}:")
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x")

                # Repair loop
                still_drifted = drifted
                for step in range(tc.max_repair_steps):
                    n_adj = repair_toward_snapshot(model, merged_snapshot, device, tc.repair_alpha)
                    phase_repair_steps += 1

                    # Re-evaluate
                    repair_ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_phases[:-1], tc.drift_threshold)

                    log_entry = {
                        "repair_step": step + 1,
                        "ppls_after": {k: f"{v:.2f}" for k, v in repair_ppls.items()},
                        "still_drifted": list(still_drifted.keys()) if still_drifted else [],
                    }
                    phase_repair_log.append(log_entry)

                    if not still_drifted:
                        print(f"  [AVR] Repair converged at step {step+1}")
                        break

                if still_drifted:
                    print(f"  [AVR] Max repair steps ({tc.max_repair_steps}) reached, drift remains on {list(still_drifted.keys())}")

                # Update merged state with repaired model
                merged_state = get_lora_state(model)
            else:
                print(f"  [AVR] No drift — repair not needed")

        total_repair_steps += phase_repair_steps
        results["repair_log"].append({
            "phase": pk, "repair_steps": phase_repair_steps,
            "details": phase_repair_log
        })

        # Final eval for this phase
        final_ppls = eval_all_domains(model, val_ds, domain_order, device, tc.eval_samples)

        # Update best PPLs
        for dpk, dppl in final_ppls.items():
            if dpk not in best_ppls or dppl < best_ppls[dpk]:
                best_ppls[dpk] = dppl

        # Snapshot current merged_state for next phase's repair target
        merged_snapshot = copy.deepcopy(merged_state)

        ev = {"perplexity": final_ppls, "avg_loss": tl / max(gs, 1),
              "repair_steps_this_phase": phase_repair_steps}
        results["phases"][pk] = ev

        print(f"  Eval (merged+AVR):")
        for p, ppl in final_ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")
        if phase_repair_steps > 0:
            print(f"  [AVR] Repair steps this phase: {phase_repair_steps}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    results["total_repair_steps"] = total_repair_steps
    print(f"\n  [AVR] Total repair steps across all phases: {total_repair_steps}")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_ff(results, domain_order):
    """Compute Forgetting Factor.
    FF(X) = PPL(X)_after_all_training / PPL(X)_right_after_training_on_X
    """
    phases = results.get("phases", {})
    ff = {}
    final_key = domain_order[-1]
    for pk in domain_order:
        if pk not in phases: continue
        after_ppl = phases[pk].get("perplexity", {}).get(pk, None)
        final_ppl = phases[final_key].get("perplexity", {}).get(pk, None)
        if after_ppl and final_ppl and after_ppl > 0:
            ff[pk] = final_ppl / after_ppl
    return ff

def compute_plasticity_cost(results, domain_order):
    """PPL on the newest domain after all training."""
    phases = results.get("phases", {})
    last_pk = domain_order[-1]
    return phases.get(last_pk, {}).get("perplexity", {}).get(last_pk, None)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_all():
    tc = TrainConfig()
    mcfg = MODEL_CONFIG
    device = tc.device

    all_results = {"config": {
        "methods": ["naive", "ewc", "slao", "slao_avr"],
        "seeds": tc.seeds,
        "forward_order": FORWARD_ORDER,
        "reverse_order": REVERSE_ORDER,
        "ewc_lambda": tc.ewc_lambda,
        "drift_threshold": tc.drift_threshold,
        "repair_alpha": tc.repair_alpha,
    }, "runs": {}}

    # ── FORWARD ORDER: all methods × all seeds ──
    for seed in tc.seeds:
        print(f"\n{'='*80}")
        print(f"  FORWARD ORDER (A→B→C) | SEED = {seed}")
        print(f"{'='*80}")

        torch.manual_seed(seed); random.seed(seed)
        _, tokenizer = _load_base(mcfg, device)
        phases_data, val_ds = {}, {}
        for pk, d in DOMAINS.items():
            t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
            phases_data[pk] = t; val_ds[pk] = v

        for method_name in ["naive", "ewc", "slao", "slao_avr"]:
            torch.manual_seed(seed); random.seed(seed)
            if method_name == "naive":
                r = run_naive(mcfg, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            elif method_name == "ewc":
                r = run_ewc(mcfg, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            elif method_name == "slao":
                r = run_slao(mcfg, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            elif method_name == "slao_avr":
                r = run_slao_avr(mcfg, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            all_results["runs"][f"{method_name}_fwd_s{seed}"] = r

    # ── COMPUTE-MATCHED NAIVE ──
    avg_repair = 0
    n_avr = 0
    for key, r in all_results["runs"].items():
        if r.get("method") == "slao_avr" and "total_repair_steps" in r:
            avg_repair += r["total_repair_steps"]
            n_avr += 1
    avg_repair = avg_repair // max(n_avr, 1)
    print(f"\n  [COMPUTE-MATCH] avg repair steps = {avg_repair}")

    if avg_repair > 0:
        for seed in tc.seeds:
            torch.manual_seed(seed); random.seed(seed)
            _, tokenizer = _load_base(mcfg, device)
            phases_data, val_ds = {}, {}
            for pk, d in DOMAINS.items():
                t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
                phases_data[pk] = t; val_ds[pk] = v
            r = run_naive(mcfg, tc, phases_data, val_ds, FORWARD_ORDER, device, seed, extra_steps=avg_repair)
            all_results["runs"][f"naive_ext_fwd_s{seed}"] = r

    # ── REVERSE ORDER: all methods × seed=42 ──
    print(f"\n{'='*80}")
    print(f"  REVERSE ORDER (C→B→A) | SEED = 42")
    print(f"{'='*80}")

    seed = 42
    torch.manual_seed(seed); random.seed(seed)
    _, tokenizer = _load_base(mcfg, device)
    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    for method_name in ["naive", "ewc", "slao", "slao_avr"]:
        torch.manual_seed(seed); random.seed(seed)
        if method_name == "naive":
            r = run_naive(mcfg, tc, phases_data, val_ds, REVERSE_ORDER, device, seed)
        elif method_name == "ewc":
            r = run_ewc(mcfg, tc, phases_data, val_ds, REVERSE_ORDER, device, seed)
        elif method_name == "slao":
            r = run_slao(mcfg, tc, phases_data, val_ds, REVERSE_ORDER, device, seed)
        elif method_name == "slao_avr":
            r = run_slao_avr(mcfg, tc, phases_data, val_ds, REVERSE_ORDER, device, seed)
        all_results["runs"][f"{method_name}_rev_s{seed}"] = r

    # ── SAVE ──
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, "v11_full.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    return all_results


def print_summary(all_results):
    tc = TrainConfig()
    print(f"\n{'='*90}")
    print(f"V11: CONTROLLED COMPARISON — NAIVE / EWC / SLAO / SLAO+AVR")
    print(f"  Same data, same seed, same model, same run")
    print(f"  EWC λ={tc.ewc_lambda} | AVR threshold={tc.drift_threshold}, α={tc.repair_alpha}")
    print(f"{'='*90}")

    # ── FORWARD ORDER: per-seed detail ──
    print(f"\n── FORWARD ORDER (A→B→C) ──")
    print(f"{'Method':<12} {'Seed':<6} {'A':<8} {'B':<8} {'C':<8} {'FF(A)':<8} {'FF(B)':<8} {'New PPL':<8} {'Repair':<7}")
    print("-" * 80)

    ff_agg = {}
    for seed in tc.seeds:
        for method in ["naive", "ewc", "slao", "slao_avr"]:
            key = f"{method}_fwd_s{seed}"
            if key not in all_results["runs"]: continue
            r = all_results["runs"][key]
            ff = compute_ff(r, FORWARD_ORDER)
            plasticity = compute_plasticity_cost(r, FORWARD_ORDER)
            phases = r.get("phases", {})
            last_ppl = phases.get(FORWARD_ORDER[-1], {}).get("perplexity", {})

            ff_a = ff.get("A", 0)
            ff_b = ff.get("B", 0)
            repair = r.get("total_repair_steps", "-")

            print(f"{method:<12} {seed:<6} {last_ppl.get('A',0):<8.1f} {last_ppl.get('B',0):<8.1f} "
                  f"{last_ppl.get('C',0):<8.1f} {ff_a:<8.2f} {ff_b:<8.2f} "
                  f"{plasticity:<8.2f} {repair}")

            if method not in ff_agg: ff_agg[method] = {"FF_A": [], "FF_B": [], "plasticity": []}
            if ff_a: ff_agg[method]["FF_A"].append(ff_a)
            if ff_b: ff_agg[method]["FF_B"].append(ff_b)
            if plasticity: ff_agg[method]["plasticity"].append(plasticity)

    # Compute-matched naive
    for seed in tc.seeds:
        key = f"naive_ext_fwd_s{seed}"
        if key not in all_results["runs"]: continue
        r = all_results["runs"][key]
        ff = compute_ff(r, FORWARD_ORDER)
        plasticity = compute_plasticity_cost(r, FORWARD_ORDER)
        phases = r.get("phases", {})
        last_ppl = phases.get(FORWARD_ORDER[-1], {}).get("perplexity", {})
        ff_a = ff.get("A", 0)
        ff_b = ff.get("B", 0)
        print(f"{'naive_ext':<12} {seed:<6} {last_ppl.get('A',0):<8.1f} {last_ppl.get('B',0):<8.1f} "
              f"{last_ppl.get('C',0):<8.1f} {ff_a:<8.2f} {ff_b:<8.2f} {plasticity:<8.2f} {'-':<7}")
        if "naive_ext" not in ff_agg: ff_agg["naive_ext"] = {"FF_A": [], "FF_B": [], "plasticity": []}
        if ff_a: ff_agg["naive_ext"]["FF_A"].append(ff_a)
        if ff_b: ff_agg["naive_ext"]["FF_B"].append(ff_b)
        if plasticity: ff_agg["naive_ext"]["plasticity"].append(plasticity)

    # ── AGGREGATED ──
    print(f"\n{'='*80}")
    print(f"{'Method':<12} {'FF(A) μ±σ':<16} {'FF(B) μ±σ':<16} {'New PPL μ±σ':<16} {'Repair total':<12}")
    print("-" * 80)
    for method in ["naive", "ewc", "slao", "slao_avr", "naive_ext"]:
        if method not in ff_agg: continue
        d = ff_agg[method]
        def fmt(vals):
            if not vals: return "N/A"
            m = sum(vals)/len(vals)
            s = (sum((x-m)**2 for x in vals)/len(vals))**0.5 if len(vals)>1 else 0
            return f"{m:.3f}±{s:.3f}"
        repair_total = sum(r.get("total_repair_steps", 0)
                          for k, r in all_results["runs"].items()
                          if r.get("method") == method and "fwd" in k)
        print(f"{method:<12} {fmt(d['FF_A']):<16} {fmt(d['FF_B']):<16} {fmt(d['plasticity']):<16} {repair_total:<12}")

    # ── REVERSE ORDER ──
    print(f"\n── REVERSE ORDER (C→B→A) — seed=42 ──")
    print(f"{'Method':<12} {'C':<8} {'B':<8} {'A':<8} {'FF(C)':<8} {'FF(B)':<8} {'Repair':<7}")
    print("-" * 55)
    for method in ["naive", "ewc", "slao", "slao_avr"]:
        key = f"{method}_rev_s42"
        if key not in all_results["runs"]: continue
        r = all_results["runs"][key]
        ff = compute_ff(r, REVERSE_ORDER)
        phases = r.get("phases", {})
        last_ppl = phases.get(REVERSE_ORDER[-1], {}).get("perplexity", {})
        repair = r.get("total_repair_steps", "-")
        print(f"{method:<12} {last_ppl.get('C',0):<8.1f} {last_ppl.get('B',0):<8.1f} "
              f"{last_ppl.get('A',0):<8.1f} {ff.get('C',0):<8.2f} {ff.get('B',0):<8.2f} {repair}")

    # ── VERDICT ──
    slao_ffa = ff_agg.get("slao", {}).get("FF_A", [])
    avr_ffa = ff_agg.get("slao_avr", {}).get("FF_A", [])
    naive_ffa = ff_agg.get("naive", {}).get("FF_A", [])

    if slao_ffa and naive_ffa:
        slao_m = sum(slao_ffa)/len(slao_ffa)
        naive_m = sum(naive_ffa)/len(naive_ffa)
        reduction = (1 - slao_m / naive_m) * 100
        print(f"\n  SLAO reduces forgetting by {reduction:.1f}% vs naive (forward order)")

    if avr_ffa and slao_ffa:
        avr_m = sum(avr_ffa)/len(avr_ffa)
        slao_m = sum(slao_ffa)/len(slao_ffa)
        if abs(avr_m - slao_m) < 0.02:
            print(f"  AVR repair is REDUNDANT on top of SLAO (FF(A) {avr_m:.3f} vs {slao_m:.3f})")
            print(f"  → SLAO's merge already handles what AVR's repair targets")
        elif avr_m < slao_m:
            print(f"  AVR COMPLEMENTS SLAO: FF(A) {avr_m:.3f} vs SLAO {slao_m:.3f}")
        else:
            print(f"  AVR HURTS SLAO: FF(A) {avr_m:.3f} vs SLAO {slao_m:.3f} (repair overcorrects)")

    ext_ffa = ff_agg.get("naive_ext", {}).get("FF_A", [])
    if ext_ffa and naive_ffa and avr_ffa:
        ext_m = sum(ext_ffa)/len(ext_ffa)
        naive_m = sum(naive_ffa)/len(naive_ffa)
        avr_m = sum(avr_ffa)/len(avr_ffa)
        if ext_m < naive_m:
            print(f"\n  Compute-matched naive: FF(A)={ext_m:.3f} vs plain naive={naive_m:.3f}")
            gap = naive_m - avr_m
            if gap > 0:
                closed = (naive_m - ext_m) / gap * 100
                print(f"  Extra training alone closes {closed:.0f}% of the naive→avr gap")

    print(f"{'='*90}")


run_all()
