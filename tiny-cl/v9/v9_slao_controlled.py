"""
V9.1: SLAO — Controlled Experiment

FIXES from V9:
1. Naive baseline IN THE SAME RUN (same data, same seed, same model)
   → "LIVING MODEL ACHIEVED!" now means something
2. Module verification: print what LoRA actually wraps
3. Multi-seed (3 seeds by default)
4. Honest naming: not "conv_only" if it hits attention too

The V9 run got FF(A)=1.09x with rank=32 — but that was compared against
hardcoded numbers from different scripts. This version earns the comparison.

SLAO Algorithm (Qiao & Mahdavi, ICLR 2026, arXiv 2512.23017):
1. Task 1: standard fine-tune → A_merge=A_1, B_merge=B_1
2. Task i (i>1):
   a. A_init = QR(prev_A)^T (orthogonal rows), B_init = prev_B
   b. Fine-tune both A and B on new task
   c. A_merge = A_ft (replace), B_merge = B_merge + λ(B_ft - B_merge)
      where λ = 1/sqrt(i)

USAGE: Copy-paste into one Kaggle cell. Run it.
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
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Both in_proj and out_proj — out_proj exists in attn layers too
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
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seeds: list = field(default_factory=lambda: [42, 123, 456])
    eval_samples: int = 1024
    results_dir: str = "v9_controlled_results"

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
# MODEL + LoRA
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

def create_model_with_lora(mcfg, device):
    """Create model with LoRA. Prints what modules are actually wrapped."""
    from peft import LoraConfig, get_peft_model, TaskType
    model, tokenizer = _load_base(mcfg, device)
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=mcfg.target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── VERIFY: what did LoRA actually wrap? ──
    from peft.tuners.lora.layer import LoraLayer
    conv_count, attn_count, other_count = 0, 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        is_conv = any(f"layers.{idx}." in name for idx in CONV_LAYER_IDS)
        is_attn = any(f"layers.{idx}." in name for idx in ATTN_LAYER_IDS)
        if is_conv: conv_count += 1
        elif is_attn: attn_count += 1
        else: other_count += 1
    print(f"  [VERIFY] LoRA wraps: {conv_count} conv, {attn_count} attn, {other_count} other = {conv_count+attn_count+other_count} total")
    if attn_count > 0:
        print(f"  [VERIFY] ⚠ {attn_count} attn layers have LoRA — this is NOT conv-only!")
    print(f"  [VERIFY] target_modules={mcfg.target_modules}")
    return model, tokenizer


# ──────────────────────────────────────────────
# SLAO CORE
# ──────────────────────────────────────────────

def extract_orthogonal_A(model):
    """Paper eq 13: QR decomposition of A^T, with sign correction."""
    from peft.tuners.lora.layer import LoraLayer
    ortho_A = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()  # [rank, in_features]
        A_T = A.T.contiguous()  # [in_features, rank]
        Q, R = torch.linalg.qr(A_T)
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        A_orth = Q.T  # [rank, in_features]
        ortho_A[name] = A_orth
    print(f"  [ORTHO-A] Extracted from {len(ortho_A)} LoRA layers")
    return ortho_A

def get_lora_state(model, device="cpu"):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state, device):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device))

def initialize_slao(model, ortho_A, prev_ft_B, device):
    from peft.tuners.lora.layer import LoraLayer
    n_init = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        if name in ortho_A:
            A_init = ortho_A[name].to(device)
            module.lora_A["default"].weight.data.copy_(A_init.to(module.lora_A["default"].weight.data.dtype))
        B_key = f"{name}.lora_B.default.weight"
        if B_key in prev_ft_B:
            B_val = prev_ft_B[B_key].to(device)
            module.lora_B["default"].weight.data.copy_(B_val.to(module.lora_B["default"].weight.data.dtype))
        n_init += 1
    print(f"  [SLAO-INIT] {n_init} layers: A=ortho(QR), B=prev_finetuned")

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


# ──────────────────────────────────────────────
# METHOD 1: NAIVE (no continual learning)
# ──────────────────────────────────────────────

def run_naive(mcfg, tc, tokenizer, phases_data, val_ds, device, seed):
    """Sequential fine-tuning with no CL — same model, same data, same seed.
    This IS the baseline that makes the comparison honest."""
    print(f"\n{'#'*70}")
    print(f"# NAIVE (no continual learning) | seed={seed}")
    print(f"{'#'*70}")

    model, _ = create_model_with_lora(mcfg, device)

    results = {"method": "naive", "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        dataset = phases_data[pk]
        # Enable all LoRA params
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        print(f"\n  Phase {pk}: {DOMAINS[pk].display_name}")
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
        gs, tl = 0, 0.0
        t0 = time.time()

        for epoch in range(tc.epochs_per_phase):
            for batch in loader:
                model.train()
                out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += out.loss.item(); gs += 1
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

        # Eval on ALL domains (not just current)
        ppls = {}
        for eval_pk in DOMAINS.keys():
            if val_ds[eval_pk] is not None:
                ppls[eval_pk] = compute_ppl(model, val_ds[eval_pk], device, tc.eval_samples)

        ev = {"perplexity": ppls, "avg_loss": tl / max(gs, 1), "time": time.time() - t0}
        results["phases"][pk] = ev

        print(f"  Eval:")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ──────────────────────────────────────────────
# METHOD 2: SLAO (Algorithm 1 from paper)
# ──────────────────────────────────────────────

def run_slao(mcfg, tc, tokenizer, phases_data, val_ds, device, seed):
    """SLAO: Single LoRA with orthogonal init and B interpolation."""
    print(f"\n{'#'*70}")
    print(f"# SLAO (ICLR 2026 Algorithm 1) | seed={seed}")
    print(f"# A=replace, B=interpolate(λ=1/√i)")
    print(f"{'#'*70}")

    model, _ = create_model_with_lora(mcfg, device)

    merged_state = None
    prev_ft_state = None
    ortho_A = None
    task_num = 0

    results = {"method": "slao", "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        task_num += 1
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

        # Train
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
        gs, tl = 0, 0.0
        t0 = time.time()

        for epoch in range(tc.epochs_per_phase):
            for batch in loader:
                model.train()
                out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += out.loss.item(); gs += 1
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

        # Save fine-tuned state
        prev_ft_state = get_lora_state(model, device)

        # Merge
        if merged_state is None:
            merged_state = prev_ft_state.copy()
            print(f"  Task {task_num}: merged = finetuned (first task)")
        else:
            merged_state = slao_merge_B(merged_state, prev_ft_state, task_num, device)

        # Load merged for eval
        set_lora_state(model, merged_state, device)

        # Eval
        ppls = {}
        for eval_pk in DOMAINS.keys():
            if val_ds[eval_pk] is not None:
                ppls[eval_pk] = compute_ppl(model, val_ds[eval_pk], device, tc.eval_samples)

        ev = {"perplexity": ppls, "avg_loss": tl / max(gs, 1), "time": time.time() - t0}
        results["phases"][pk] = ev

        print(f"  Eval (merged LoRA):")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


# ──────────────────────────────────────────────
# COMPUTE FF METRICS
# ──────────────────────────────────────────────

def compute_ff(results):
    """Compute Forgetting Factor from a results dict.
    FF(A) = PPL_A_after_C / PPL_A_after_A
    FF(B) = PPL_B_after_C / PPL_B_after_B
    """
    phases = results.get("phases", {})
    keys = list(phases.keys())
    if len(keys) < 3:
        return {"FF_A": None, "FF_B": None}

    final_ppl = phases[keys[-1]].get("perplexity", {})
    ppl_a_after_a = phases["A"].get("perplexity", {}).get("A", None)
    ppl_b_after_b = phases["B"].get("perplexity", {}).get("B", None)
    ppl_a_final = final_ppl.get("A", None)
    ppl_b_final = final_ppl.get("B", None)

    ff_a = ppl_a_final / ppl_a_after_a if (ppl_a_after_a and ppl_a_final and ppl_a_after_a > 0) else None
    ff_b = ppl_b_final / ppl_b_after_b if (ppl_b_after_b and ppl_b_final and ppl_b_after_b > 0) else None

    return {"FF_A": ff_a, "FF_B": ff_b}


# ──────────────────────────────────────────────
# MAIN: RUN BOTH METHODS × MULTIPLE SEEDS
# ──────────────────────────────────────────────

def run_all():
    tc = TrainConfig()
    mcfg = MODEL_CONFIG
    device = tc.device

    all_results = {"methods": ["naive", "slao"], "seeds": tc.seeds, "runs": {}}

    for seed in tc.seeds:
        print(f"\n{'='*80}")
        print(f"  SEED = {seed}")
        print(f"{'='*80}")

        torch.manual_seed(seed); random.seed(seed)

        # Prepare data ONCE per seed — same for both methods
        _, tokenizer = _load_base(mcfg, device)
        phases_data, val_ds = {}, {}
        for pk, d in DOMAINS.items():
            t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
            phases_data[pk] = t; val_ds[pk] = v

        # ── NAIVE ──
        torch.manual_seed(seed); random.seed(seed)
        naive_results = run_naive(mcfg, tc, tokenizer, phases_data, val_ds, device, seed)
        all_results["runs"][f"naive_seed{seed}"] = naive_results

        # ── SLAO ──
        torch.manual_seed(seed); random.seed(seed)
        slao_results = run_slao(mcfg, tc, tokenizer, phases_data, val_ds, device, seed)
        all_results["runs"][f"slao_seed{seed}"] = slao_results

    # ── SUMMARY ──
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, "controlled_comparison.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    return all_results


def print_summary(all_results):
    print(f"\n{'='*80}")
    print(f"V9.1: CONTROLLED COMPARISON — NAIVE vs SLAO")
    print(f"  Same data, same seed, same model, same run")
    print(f"  target_modules={MODEL_CONFIG.target_modules} (conv+attn)")
    print(f"{'='*80}")

    # Aggregate FF across seeds
    ff_data = {"naive": {"FF_A": [], "FF_B": []}, "slao": {"FF_A": [], "FF_B": []}}
    ppl_data = {"naive": [], "slao": []}

    for run_key, run_results in all_results["runs"].items():
        method = run_results["method"]
        ff = compute_ff(run_results)
        if ff["FF_A"] is not None:
            ff_data[method]["FF_A"].append(ff["FF_A"])
        if ff["FF_B"] is not None:
            ff_data[method]["FF_B"].append(ff["FF_B"])

        # Final PPLs
        phases = run_results.get("phases", {})
        keys = list(phases.keys())
        if len(keys) >= 3:
            fp = phases[keys[-1]].get("perplexity", {})
            ppl_data[method].append({
                "A": fp.get("A", 0), "B": fp.get("B", 0), "C": fp.get("C", 0),
                "seed": run_results.get("seed", "?")
            })

    # Print per-seed details
    print(f"\n{'Method':<10} {'Seed':<8} {'A PPL':<10} {'B PPL':<10} {'C PPL':<10} {'FF(A)':<10} {'FF(B)':<10}")
    print("-" * 68)
    for run_key, run_results in all_results["runs"].items():
        method = run_results["method"]
        seed = run_results.get("seed", "?")
        phases = run_results.get("phases", {})
        keys = list(phases.keys())
        if len(keys) < 3: continue
        fp = phases[keys[-1]].get("perplexity", {})
        ff = compute_ff(run_results)
        ff_a_str = f"{ff['FF_A']:.2f}x" if ff["FF_A"] else "N/A"
        ff_b_str = f"{ff['FF_B']:.2f}x" if ff["FF_B"] else "N/A"
        print(f"{method:<10} {seed:<8} {fp.get('A',0):<10.1f} {fp.get('B',0):<10.1f} {fp.get('C',0):<10.1f} {ff_a_str:<10} {ff_b_str:<10}")

    # Print aggregated
    print(f"\n{'='*68}")
    print(f"{'Method':<10} {'FF(A) mean':<14} {'FF(A) std':<14} {'FF(B) mean':<14} {'FF(B) std':<14}")
    print("-" * 68)
    for method in ["naive", "slao"]:
        ff_as = ff_data[method]["FF_A"]
        ff_bs = ff_data[method]["FF_B"]
        if ff_as:
            mean_a = sum(ff_as) / len(ff_as)
            std_a = (sum((x - mean_a)**2 for x in ff_as) / len(ff_as)) ** 0.5 if len(ff_as) > 1 else 0
            mean_b = sum(ff_bs) / len(ff_bs) if ff_bs else 0
            std_b = (sum((x - mean_b)**2 for x in ff_bs) / len(ff_bs)) ** 0.5 if len(ff_bs) > 1 else 0
            print(f"{method:<10} {mean_a:<14.3f} {std_a:<14.3f} {mean_b:<14.3f} {std_b:<14.3f}")

    # Verdict
    slao_ffa = ff_data["slao"]["FF_A"]
    naive_ffa = ff_data["naive"]["FF_A"]
    if slao_ffa and naive_ffa:
        slao_mean = sum(slao_ffa) / len(slao_ffa)
        naive_mean = sum(naive_ffa) / len(naive_ffa)
        reduction = (1 - slao_mean / naive_mean) * 100
        print(f"\n  SLAO reduces forgetting by {reduction:.1f}% vs naive (same-run comparison)")
        if slao_mean < 1.1:
            print(f"  >> LIVING MODEL ACHIEVED! (FF(A)={slao_mean:.2f}x < 1.10x)")
        elif slao_mean < 1.2:
            print(f"  >> Near-living model (FF(A)={slao_mean:.2f}x)")
        else:
            print(f"  >> Still forgetting (FF(A)={slao_mean:.2f}x)")

    print(f"{'='*80}")


run_all()
