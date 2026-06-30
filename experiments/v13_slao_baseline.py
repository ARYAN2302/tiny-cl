"""
V13: The Full Picture — Where Are We?

Methods:
  1. naive      — no CL baseline
  2. slao       — SLAO rank=32 (current best, reproducibility check)
  3. fixed_a    — freeze A after task 1, train B only, merge B
                  (cheapest diagnostic: if gap closes → cross-term is the problem)
  4. dw_stitch  — ΔW-stitch: merge in ΔW space, memory-efficient SVD recompress
                  (main contribution candidate, uses QR+2r×2r SVD trick)
  5. slao_r64   — SLAO rank=64 (engineering baseline)

Domain orderings (seed=42 only):
  - Forward:  A→B→C
  - Reverse:  C→B→A
  - Random 1: B→C→A
  - Random 2: A→C→B
  - Random 3: C→A→B

Diagnostics:
  - Cross-term noise ratio: ‖Σ_{i≠j} B_i@A_j‖_F / ‖Σ_i B_i@A_i‖_F
  - Plasticity cost (newest domain PPL)
  - BWT (backward transfer)

Based on second opinion feedback:
  - Fixed-A test first (mathematically eliminates cross-term noise)
  - Memory-efficient ΔW-stitch (QR + 2r×2r SVD, no OOM)
  - Multiple domain orderings (test curriculum bias)
  - Cross-term noise measurement (validate diagnosis)

USAGE: Copy-paste into one Kaggle cell. ~4-5 hours on T4.
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
RAND_ORDERS = [
    (["B", "C", "A"], "B→C→A"),
    (["A", "C", "B"], "A→C→B"),
    (["C", "A", "B"], "C→A→B"),
]
ALL_SEEDS = [42, 123, 456]
ORDER_SEED = 42

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    epochs_per_phase: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: list = field(default_factory=lambda: ALL_SEEDS)
    eval_samples: int = 1024
    results_dir: str = "v13_results"

DOMAINS = {
    "A": {"name": "medical", "display": "Medical",
           "dataset": "epfl-llm/guidelines", "field": "clean_text"},
    "B": {"name": "code", "display": "Code",
           "dataset": "iamtarun/python_code_instructions_18k_alpaca", "field": "output"},
    "C": {"name": "creative", "display": "Creative",
           "dataset": "roneneldan/TinyStories", "field": "text"},
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

def prepare_domain(domain_key, tokenizer, context_length, max_tokens=1_000_000, seed=42):
    from datasets import load_dataset
    d = DOMAINS[domain_key]
    print(f"  Loading: {d['display']}")
    ds = load_dataset(path=d["dataset"], split="train")
    texts = [t for t in ds[d["field"]] if t and len(t.strip()) > 10]
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
    return TextDataset(token_ids[:n_train], context_length), \
           TextDataset(token_ids[n_train:n_train + n_val], context_length)


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

def _load_base(hf_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading {hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    return model, tokenizer

def create_model(rank, device, hf_id="LiquidAI/LFM2.5-350M"):
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(hf_id, device)
    lora_config = LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.05,
        target_modules=["in_proj", "out_proj"], bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    from peft.tuners.lora.layer import LoraLayer
    conv_c, attn_c = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if any(f"layers.{idx}." in name for idx in CONV_LAYER_IDS): conv_c += 1
        elif any(f"layers.{idx}." in name for idx in ATTN_LAYER_IDS): attn_c += 1
    print(f"  [VERIFY] LoRA: {conv_c} conv + {attn_c} attn = {conv_c+attn_c} total (rank={rank})")
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

def train_phase(model, dataset, tc, device, trainable_filter=None):
    for n, p in model.named_parameters():
        if trainable_filter:
            p.requires_grad = trainable_filter(n)
        elif "lora_" in n:
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    for epoch in range(tc.epochs_per_phase):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")
    return gs, tl

def eval_all(model, val_ds, domain_order, device, eval_samples):
    ppls = {}
    for pk in domain_order:
        if val_ds.get(pk) is not None:
            ppls[pk] = compute_ppl(model, val_ds[pk], device, eval_samples)
    return ppls


# ──────────────────────────────────────────────
# SLAO CORE
# ──────────────────────────────────────────────

def slao_extract_ortho_A(model):
    from peft.tuners.lora.layer import LoraLayer
    ortho_A = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()
        Q, R = torch.linalg.qr(A.T.contiguous())
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        ortho_A[name] = Q.T
    return ortho_A

def slao_init(model, ortho_A, prev_ft_B, device):
    from peft.tuners.lora.layer import LoraLayer
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

def slao_merge(merged_state, ft_state, task_num, device):
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
# ΔW-STITCH CORE — Memory-Efficient SVD Merge
# ──────────────────────────────────────────────

def dw_stitch_merge(merged_state, ft_state, task_num, device):
    """Merge in ΔW space using memory-efficient QR + tiny SVD trick.
    
    B_cat = [B_old, λ·B_new]    shape: [d_out, 2r]
    A_cat = [A_old; A_new]       shape: [2r, d_in]
    
    B_cat @ A_cat = B_old@A_old + λ·B_new@A_new  (exact ΔW-stitch formula)
    
    QR decomposition reduces SVD to a 2r×2r matrix.
    SVD finds optimal rank-r approximation.
    """
    lam = 1.0 / math.sqrt(task_num)
    new_merged = {}
    n_layers = 0
    
    a_keys = sorted([k for k in ft_state if "lora_A" in k])
    
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in ft_state or a_key not in merged_state or b_key not in merged_state:
            new_merged[a_key] = ft_state[a_key].cpu().clone()
            if b_key in ft_state:
                new_merged[b_key] = ft_state[b_key].cpu().clone()
            continue
        
        A_old = merged_state[a_key].float().to(device)
        B_old = merged_state[b_key].float().to(device)
        A_new = ft_state[a_key].float().to(device)
        B_new = ft_state[b_key].float().to(device)
        r = A_old.shape[0]
        
        # Concatenate
        B_cat = torch.cat([B_old, lam * B_new], dim=1)  # [d_out, 2r]
        A_cat = torch.cat([A_old, A_new], dim=0)          # [2r, d_in]
        
        # QR decomposition (only touches 2r dimension!)
        Q_B, R_B = torch.linalg.qr(B_cat)     # Q_B: [d_out, 2r], R_B: [2r, 2r]
        Q_A, R_A = torch.linalg.qr(A_cat.T)    # Q_A: [d_in, 2r],  R_A: [2r, 2r]
        L_A = R_A.T                              # [2r, 2r]
        
        # Tiny interaction matrix
        M = R_B @ L_A  # [2r, 2r] — e.g. 64×64 for rank=32
        
        # SVD on tiny matrix
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        
        # Truncate to rank r
        U_r = U[:, :r]
        S_r = S[:r]
        Vh_r = Vh[:r, :]
        
        # Reconstruct
        sqrt_S = torch.sqrt(S_r)
        B_merged = Q_B @ U_r @ torch.diag(sqrt_S)   # [d_out, r]
        A_merged = torch.diag(sqrt_S) @ Vh_r @ Q_A.T # [r, d_in]
        
        orig_dtype = ft_state[a_key].dtype
        new_merged[a_key] = A_merged.to(orig_dtype).cpu().clone()
        new_merged[b_key] = B_merged.to(orig_dtype).cpu().clone()
        n_layers += 1
    
    for key in ft_state:
        if key not in new_merged:
            new_merged[key] = ft_state[key].cpu().clone()
    
    print(f"  [ΔW-STITCH] Task {task_num}: {n_layers} layers merged via QR+SVD (λ={lam:.4f}, SVD size={2*r}×{2*r})")
    return new_merged


# ──────────────────────────────────────────────
# CROSS-TERM NOISE MEASUREMENT
# ──────────────────────────────────────────────

def measure_cross_term_noise(per_task_states, device):
    """noise_ratio = ‖Σ_{i≠j} B_i@A_j‖_F / ‖Σ_i B_i@A_i‖_F"""
    a_keys = sorted([k for k in per_task_states[0] if "lora_A" in k])
    total_signal_sq = 0.0
    total_noise_sq = 0.0
    
    for a_key in a_keys:
        b_key = a_key.replace("lora_A", "lora_B")
        if b_key not in per_task_states[0]: continue
        
        n_tasks = len(per_task_states)
        As = [per_task_states[i][a_key].float().to(device) for i in range(n_tasks)]
        Bs = [per_task_states[i][b_key].float().to(device) for i in range(n_tasks)]
        
        d_out = Bs[0].shape[0]
        d_in = As[0].shape[1]
        signal = torch.zeros(d_out, d_in, device=device)
        noise = torch.zeros(d_out, d_in, device=device)
        
        for i in range(n_tasks):
            signal += Bs[i] @ As[i]
            for j in range(n_tasks):
                if i != j:
                    noise += Bs[i] @ As[j]
        
        total_signal_sq += signal.norm().item() ** 2
        total_noise_sq += noise.norm().item() ** 2
        
        del As, Bs, signal, noise
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    signal_mag = math.sqrt(total_signal_sq) if total_signal_sq > 0 else 1e-10
    noise_mag = math.sqrt(total_noise_sq)
    return noise_mag / signal_mag


# ──────────────────────────────────────────────
# METHOD RUNNERS
# ──────────────────────────────────────────────

def run_naive(rank, tc, phases_data, val_ds, domain_order, device, seed):
    print(f"\n{'#'*70}\n# NAIVE rank={rank} | seed={seed} | {'→'.join(domain_order)}\n{'#'*70}")
    model, _ = create_model(rank, device)
    results = {"method": "naive", "rank": rank, "seed": seed, "domain_order": domain_order, "phases": {}}
    per_task_states = []
    
    for task_num, pk in enumerate(domain_order, 1):
        gs, tl = train_phase(model, phases_data[pk], tc, device)
        per_task_states.append(get_lora_state(model))
        ppls = eval_all(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval: " + " | ".join(f"{p}: {v:.2f}" for p, v in ppls.items()))
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    results["cross_term_noise"] = measure_cross_term_noise(per_task_states, device)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results


def run_slao(rank, tc, phases_data, val_ds, domain_order, device, seed):
    print(f"\n{'#'*70}\n# SLAO rank={rank} | seed={seed} | {'→'.join(domain_order)}\n{'#'*70}")
    model, _ = create_model(rank, device)
    merged_state = None
    prev_ft_state = None
    per_task_states = []
    results = {"method": "slao", "rank": rank, "seed": seed, "domain_order": domain_order, "phases": {}}
    
    for task_num, pk in enumerate(domain_order, 1):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True
        
        if task_num > 1:
            ortho_A = slao_extract_ortho_A(model)
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            slao_init(model, ortho_A, prev_ft_B, device)
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — SLAO init")
        else:
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — standard fine-tune")
        
        gs, tl = train_phase(model, phases_data[pk], tc, device)
        prev_ft_state = get_lora_state(model)
        per_task_states.append(copy.deepcopy(prev_ft_state))
        
        if merged_state is None:
            merged_state = prev_ft_state.copy()
        else:
            merged_state = slao_merge(merged_state, prev_ft_state, task_num, device)
        
        set_lora_state(model, merged_state, device)
        ppls = eval_all(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval: " + " | ".join(f"{p}: {v:.2f}" for p, v in ppls.items()))
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    results["cross_term_noise"] = measure_cross_term_noise(per_task_states, device)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results


def run_fixed_a(rank, tc, phases_data, val_ds, domain_order, device, seed):
    """Fixed-A: train A+B on task 1, then freeze A, train B only for tasks 2+.
    
    Since A never changes: ΔW = B @ A_fixed is perfectly additive.
    Zero cross-term noise by construction.
    """
    print(f"\n{'#'*70}\n# FIXED-A rank={rank} | seed={seed} | {'→'.join(domain_order)}\n{'#'*70}")
    model, _ = create_model(rank, device)
    merged_state = None
    prev_ft_state = None
    per_task_states = []
    results = {"method": "fixed_a", "rank": rank, "seed": seed, "domain_order": domain_order, "phases": {}}
    
    for task_num, pk in enumerate(domain_order, 1):
        if task_num == 1:
            # Task 1: train both A and B
            for n, p in model.named_parameters():
                if "lora_" in n: p.requires_grad = True
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — train A+B (task 1)")
        else:
            # Task 2+: load merged B, freeze A, train B only
            if merged_state is not None:
                for n, p in model.named_parameters():
                    if "lora_B" in n and n in merged_state:
                        p.data.copy_(merged_state[n].to(device))
            for n, p in model.named_parameters():
                if "lora_A" in n: p.requires_grad = False
                elif "lora_B" in n: p.requires_grad = True
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — A FROZEN, train B only")
        
        gs, tl = train_phase(model, phases_data[pk], tc, device)
        prev_ft_state = get_lora_state(model)
        per_task_states.append(copy.deepcopy(prev_ft_state))
        
        if merged_state is None:
            merged_state = prev_ft_state.copy()
        else:
            merged_state = slao_merge(merged_state, prev_ft_state, task_num, device)
        
        set_lora_state(model, merged_state, device)
        ppls = eval_all(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval: " + " | ".join(f"{p}: {v:.2f}" for p, v in ppls.items()))
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    results["cross_term_noise"] = measure_cross_term_noise(per_task_states, device)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results


def run_dw_stitch(rank, tc, phases_data, val_ds, domain_order, device, seed):
    """ΔW-stitch: SLAO init + ΔW-space merge with memory-efficient SVD."""
    print(f"\n{'#'*70}\n# ΔW-STITCH rank={rank} | seed={seed} | {'→'.join(domain_order)}\n{'#'*70}")
    model, _ = create_model(rank, device)
    merged_state = None
    prev_ft_state = None
    per_task_states = []
    results = {"method": "dw_stitch", "rank": rank, "seed": seed, "domain_order": domain_order, "phases": {}}
    
    for task_num, pk in enumerate(domain_order, 1):
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True
        
        if task_num > 1:
            ortho_A = slao_extract_ortho_A(model)
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            slao_init(model, ortho_A, prev_ft_B, device)
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — SLAO init + ΔW-stitch merge")
        else:
            print(f"  Phase {pk}: {DOMAINS[pk]['display']} — standard fine-tune")
        
        gs, tl = train_phase(model, phases_data[pk], tc, device)
        prev_ft_state = get_lora_state(model)
        per_task_states.append(copy.deepcopy(prev_ft_state))
        
        if merged_state is None:
            merged_state = prev_ft_state.copy()
        else:
            merged_state = dw_stitch_merge(merged_state, prev_ft_state, task_num, device)
        
        set_lora_state(model, merged_state, device)
        ppls = eval_all(model, val_ds, domain_order, device, tc.eval_samples)
        results["phases"][pk] = {"perplexity": ppls, "avg_loss": tl / max(gs, 1)}
        print(f"  Eval: " + " | ".join(f"{p}: {v:.2f}" for p, v in ppls.items()))
        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()
    
    results["cross_term_noise"] = measure_cross_term_noise(per_task_states, device)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results


# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_ff(results, domain_order):
    phases = results.get("phases", {})
    ff = {}
    final_key = domain_order[-1]
    for pk in domain_order:
        after_ppl = phases.get(pk, {}).get("perplexity", {}).get(pk)
        final_ppl = phases.get(final_key, {}).get("perplexity", {}).get(pk)
        if after_ppl and final_ppl and after_ppl > 0:
            ff[pk] = final_ppl / after_ppl
    return ff


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_all():
    tc = TrainConfig()
    device = tc.device
    all_results = {"runs": {}}
    
    # ═══════════════════════════════════════════
    # PHASE 1: Forward order × 3 seeds × 5 methods
    # ═══════════════════════════════════════════
    for seed in ALL_SEEDS:
        print(f"\n{'='*80}\n  FORWARD ORDER (A→B→C) | SEED = {seed}\n{'='*80}")
        torch.manual_seed(seed); random.seed(seed)
        _, tokenizer = _load_base("LiquidAI/LFM2.5-350M", device)
        phases_data, val_ds = {}, {}
        for pk in DOMAINS:
            t, v = prepare_domain(pk, tokenizer, 512, 1_000_000, seed)
            phases_data[pk] = t; val_ds[pk] = v
        
        for method_name in ["naive", "slao", "fixed_a", "dw_stitch", "slao_r64"]:
            torch.manual_seed(seed); random.seed(seed)
            rank = 64 if method_name == "slao_r64" else 32
            
            if method_name == "naive":
                r = run_naive(rank, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            elif method_name in ("slao", "slao_r64"):
                r = run_slao(rank, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
                r["method"] = method_name
            elif method_name == "fixed_a":
                r = run_fixed_a(rank, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            elif method_name == "dw_stitch":
                r = run_dw_stitch(rank, tc, phases_data, val_ds, FORWARD_ORDER, device, seed)
            
            all_results["runs"][f"{method_name}_fwd_s{seed}"] = r
    
    # ═══════════════════════════════════════════
    # PHASE 2: Domain ordering × seed=42 × key methods
    # ═══════════════════════════════════════════
    seed = ORDER_SEED
    torch.manual_seed(seed); random.seed(seed)
    _, tokenizer = _load_base("LiquidAI/LFM2.5-350M", device)
    phases_data, val_ds = {}, {}
    for pk in DOMAINS:
        t, v = prepare_domain(pk, tokenizer, 512, 1_000_000, seed)
        phases_data[pk] = t; val_ds[pk] = v
    
    all_orderings = [(FORWARD_ORDER, "A→B→C"), (REVERSE_ORDER, "C→B→A")] + RAND_ORDERS
    
    for order, label in all_orderings:
        if order == FORWARD_ORDER:
            continue  # already have forward results
        
        print(f"\n{'='*80}\n  ORDER {label} | SEED = {seed}\n{'='*80}")
        
        for method_name in ["slao", "fixed_a", "dw_stitch"]:
            torch.manual_seed(seed); random.seed(seed)
            
            if method_name == "slao":
                r = run_slao(32, tc, phases_data, val_ds, order, device, seed)
            elif method_name == "fixed_a":
                r = run_fixed_a(32, tc, phases_data, val_ds, order, device, seed)
            elif method_name == "dw_stitch":
                r = run_dw_stitch(32, tc, phases_data, val_ds, order, device, seed)
            
            all_results["runs"][f"{method_name}_{label}_s{seed}"] = r
    
    # ═══════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, "v13_full.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    print_summary(all_results)
    return all_results


def print_summary(all_results):
    print(f"\n{'='*100}")
    print(f"V13: THE FULL PICTURE — WHERE ARE WE?")
    print(f"{'='*100}")
    
    # ── FORWARD ORDER ──
    print(f"\n{'─'*100}")
    print(f"FORWARD ORDER (A→B→C) — Multi-seed comparison")
    print(f"{'─'*100}")
    print(f"{'Method':<14} {'Rank':<5} {'Seed':<6} {'A PPL':<8} {'B PPL':<8} {'C PPL':<8} "
          f"{'FF(A)':<8} {'FF(B)':<8} {'New PPL':<8} {'Noise%':<8}")
    print("-" * 100)
    
    ff_agg = {}
    noise_agg = {}
    
    for seed in ALL_SEEDS:
        for method_name in ["naive", "slao", "fixed_a", "dw_stitch", "slao_r64"]:
            key = f"{method_name}_fwd_s{seed}"
            if key not in all_results["runs"]: continue
            r = all_results["runs"][key]
            ff = compute_ff(r, FORWARD_ORDER)
            phases = r.get("phases", {})
            last_ppl = phases.get(FORWARD_ORDER[-1], {}).get("perplexity", {})
            plasticity = last_ppl.get(FORWARD_ORDER[-1], 0)
            noise = r.get("cross_term_noise", 0)
            rank = r.get("rank", 32)
            
            ff_a = ff.get("A", 0)
            ff_b = ff.get("B", 0)
            
            print(f"{method_name:<14} {rank:<5} {seed:<6} {last_ppl.get('A',0):<8.1f} "
                  f"{last_ppl.get('B',0):<8.1f} {last_ppl.get('C',0):<8.1f} "
                  f"{ff_a:<8.2f} {ff_b:<8.2f} {plasticity:<8.2f} {noise:<8.1%}")
            
            if method_name not in ff_agg:
                ff_agg[method_name] = {"FF_A": [], "FF_B": [], "plasticity": []}
                noise_agg[method_name] = []
            if ff_a: ff_agg[method_name]["FF_A"].append(ff_a)
            if ff_b: ff_agg[method_name]["FF_B"].append(ff_b)
            if plasticity: ff_agg[method_name]["plasticity"].append(plasticity)
            if noise: noise_agg[method_name].append(noise)
    
    # ── AGGREGATED ──
    print(f"\n{'='*100}")
    print(f"AGGREGATED (3 seeds)")
    print(f"{'='*100}")
    print(f"{'Method':<14} {'FF(A) μ±σ':<18} {'FF(B) μ±σ':<18} {'New PPL μ±σ':<18} {'Noise% μ':<10}")
    print("-" * 80)
    
    for method_name in ["naive", "slao", "fixed_a", "dw_stitch", "slao_r64"]:
        if method_name not in ff_agg: continue
        d = ff_agg[method_name]
        def fmt(vals):
            if not vals: return "N/A"
            m = sum(vals)/len(vals)
            s = (sum((x-m)**2 for x in vals)/len(vals))**0.5 if len(vals)>1 else 0
            return f"{m:.3f}±{s:.3f}"
        noise_m = sum(noise_agg.get(method_name, [])) / max(len(noise_agg.get(method_name, [])), 1)
        print(f"{method_name:<14} {fmt(d['FF_A']):<18} {fmt(d['FF_B']):<18} "
              f"{fmt(d['plasticity']):<18} {noise_m:<10.1%}")
    
    # ── DOMAIN ORDERING ──
    print(f"\n{'─'*100}")
    print(f"DOMAIN ORDERING (seed=42) — Curriculum bias test")
    print(f"{'─'*100}")
    
    all_order_labels = ["A→B→C", "C→B→A", "B→C→A", "A→C→B", "C→A→B"]
    all_order_lists = [FORWARD_ORDER, REVERSE_ORDER] + [ro for ro, _ in RAND_ORDERS]
    
    for method_name in ["slao", "fixed_a", "dw_stitch"]:
        print(f"\n  {method_name}:")
        print(f"  {'Order':<10} {'FF(first)':<12} {'FF(second)':<12} {'Noise%':<10}")
        print(f"  {'-'*44}")
        
        for order, label in zip(all_order_lists, all_order_labels):
            key = f"{method_name}_{label}_s{ORDER_SEED}"
            if key not in all_results["runs"]:
                if order == FORWARD_ORDER:
                    key = f"{method_name}_fwd_s{ORDER_SEED}"
            if key not in all_results["runs"]: continue
            
            r = all_results["runs"][key]
            ff = compute_ff(r, order)
            noise = r.get("cross_term_noise", 0)
            ff_first = ff.get(order[0], 0)
            ff_second = ff.get(order[1], 0)
            print(f"  {label:<10} {ff_first:<12.2f} {ff_second:<12.2f} {noise:<10.1%}")
    
    # ── CROSS-TERM NOISE ──
    print(f"\n{'─'*100}")
    print(f"CROSS-TERM NOISE DIAGNOSIS (forward order)")
    print(f"{'─'*100}")
    for method_name in ["naive", "slao", "fixed_a", "dw_stitch", "slao_r64"]:
        if method_name not in noise_agg or not noise_agg[method_name]: continue
        avg_noise = sum(noise_agg[method_name]) / len(noise_agg[method_name])
        print(f"  {method_name:<14}: noise/signal = {avg_noise:.1%}")
    
    # ── VERDICT ──
    print(f"\n{'='*100}")
    print(f"VERDICT")
    print(f"{'='*100}")
    
    slao_ffa = ff_agg.get("slao", {}).get("FF_A", [])
    fixed_ffa = ff_agg.get("fixed_a", {}).get("FF_A", [])
    dw_ffa = ff_agg.get("dw_stitch", {}).get("FF_A", [])
    r64_ffa = ff_agg.get("slao_r64", {}).get("FF_A", [])
    
    if slao_ffa and fixed_ffa:
        slao_m = sum(slao_ffa)/len(slao_ffa)
        fixed_m = sum(fixed_ffa)/len(fixed_ffa)
        gap_closed = (slao_m - fixed_m) / slao_m * 100 if slao_m > 0 else 0
        print(f"  Fixed-A vs SLAO: FF(A) {fixed_m:.3f} vs {slao_m:.3f} (closed {gap_closed:.0f}% of gap)")
        if fixed_m < 1.05:
            print(f"  >> FIXED-A ACHIEVES LIVING MODEL! Cross-term noise was THE bottleneck.")
        elif fixed_m < slao_m - 0.02:
            print(f"  >> Fixed-A helps significantly. Cross-term noise is A major factor.")
        else:
            print(f"  >> Fixed-A barely helps. B interpolation is also losing info.")
    
    if slao_ffa and dw_ffa:
        slao_m = sum(slao_ffa)/len(slao_ffa)
        dw_m = sum(dw_ffa)/len(dw_ffa)
        gap_closed = (slao_m - dw_m) / slao_m * 100 if slao_m > 0 else 0
        print(f"  ΔW-stitch vs SLAO: FF(A) {dw_m:.3f} vs {slao_m:.3f} (closed {gap_closed:.0f}%)")
        if dw_m < 1.05:
            print(f"  >> ΔW-STITCH ACHIEVES LIVING MODEL!")
        elif dw_m < slao_m - 0.02:
            print(f"  >> ΔW-stitch helps. Optimal SVD recompression > A-replace+B-interpolate.")
        else:
            print(f"  >> ΔW-stitch doesn't help much. Capacity may be the real limit.")
    
    if r64_ffa:
        r64_m = sum(r64_ffa)/len(r64_ffa)
        print(f"  SLAO rank=64: FF(A) = {r64_m:.3f}")
        if r64_m < 1.05:
            print(f"  >> rank=64 SLAO ACHIEVES LIVING MODEL (engineering path)")
        else:
            print(f"  >> Even rank=64 doesn't reach <1.05x. Merge formula is the bottleneck.")
    
    if fixed_ffa and r64_ffa:
        fixed_m = sum(fixed_ffa)/len(fixed_ffa)
        r64_m = sum(r64_ffa)/len(r64_ffa)
        if fixed_m < r64_m:
            print(f"\n  >> DIAGNOSIS: Merge formula > Capacity (Fixed-A@r32 beats SLAO@r64)")
        elif r64_m < fixed_m:
            print(f"\n  >> DIAGNOSIS: Capacity > Merge formula (SLAO@r64 beats Fixed-A@r32)")
        else:
            print(f"\n  >> DIAGNOSIS: Both matter equally")
    
    print(f"{'='*100}")


run_all()