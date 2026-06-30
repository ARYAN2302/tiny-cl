"""
V6 Diagnostic: Where is the forgetting coming from?

After accumulate training, domain C's LoRA is still ACTIVE during eval.
This LoRA modifies the forward pass for ALL domains — it could be the
primary source of interference, not base weight corruption.

This script:
1. Runs the accumulate method (same as v6_kaggle)
2. After all 3 domains, evaluates WITH active LoRA (current V6 behavior)
3. Merges the final LoRA into base (final commit)
4. Evaluates WITH ZERO active LoRA (pure base weights)

If "no LoRA" eval gives good PPL on A/B → forgetting is from LoRA interference
If "no LoRA" eval still has bad PPL on A/B → forgetting is from base weight corruption

This tells us what to fix in V7.

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
    print(f"  [COMMIT] Merged {merged_count} conv LoRA into base, re-init for next domain")
    return merged_count


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
# DIAGNOSTIC
# ──────────────────────────────────────────────

def run_diagnostic(seed=42):
    torch.manual_seed(seed); random.seed(seed)
    mcfg = ModelConfig(); tc = TrainConfig(); device = tc.device

    print("=" * 70)
    print("V6 DIAGNOSTIC: Accumulate + Final Commit")
    print("Where is the forgetting? LoRA interference vs base corruption")
    print("=" * 70)

    model, tokenizer = create_model_conv_only(mcfg, device)

    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    phase_keys = list(DOMAINS.keys())
    initial_ppls = {}

    for pk in phase_keys:
        dataset = phases_data[pk]
        for n, p in model.named_parameters():
            if "lora_" in n and is_conv_module(n): p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(dataset, batch_size=mcfg.batch_size, shuffle=True, drop_last=False)

        print(f"\n  Phase {pk}: {DOMAINS[pk].display_name}")
        gs, tl = 0, 0.0
        for epoch in range(tc.epochs_per_phase):
            for batch in loader:
                model.train()
                out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
                opt.zero_grad(); out.loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
                opt.step(); tl += out.loss.item(); gs += 1
                if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

        # Record PPL right after training (before commit)
        ppl_after_train = compute_ppl(model, val_ds[pk], device, 512)
        initial_ppls[pk] = ppl_after_train
        print(f"    PPL on {pk} right after training: {ppl_after_train:.2f}")

        merge_conv_lora_into_base(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)

    # ─── EVAL 1: With active LoRA (current V6 behavior) ───
    print("\n" + "=" * 70)
    print("EVAL 1: With active LoRA (current V6 behavior)")
    print("  Domain C's LoRA is still active, modifying forward pass for all domains")
    print("=" * 70)
    with_lora = {}
    for pk in phase_keys:
        ppl = compute_ppl(model, val_ds[pk], device, tc.eval_samples)
        with_lora[pk] = ppl
        ff = ppl / initial_ppls[pk] if initial_ppls[pk] > 0 else 0
        print(f"  {pk}: PPL={ppl:.2f}  (initial={initial_ppls[pk]:.2f}, FF={ff:.2f}x)")

    # ─── FINAL COMMIT: merge remaining LoRA into base ───
    print("\n  [FINAL COMMIT] Merging remaining LoRA into base weights...")
    n_merged = merge_conv_lora_into_base(model)
    print(f"  Merged {n_merged} LoRA layers")

    # Verify LoRA is zero
    lora_norm = 0.0
    for name, module in model.named_modules():
        if hasattr(module, 'lora_B') and 'default' in module.lora_B:
            lora_norm += module.lora_B['default'].weight.data.norm().item()
    print(f"  LoRA B norm (should be ~0): {lora_norm:.6f}")

    # ─── EVAL 2: Without active LoRA (pure base) ───
    print("\n" + "=" * 70)
    print("EVAL 2: With ZERO active LoRA (pure accumulated base weights)")
    print("  No LoRA interference. This is the true base weight quality.")
    print("=" * 70)
    without_lora = {}
    for pk in phase_keys:
        ppl = compute_ppl(model, val_ds[pk], device, tc.eval_samples)
        without_lora[pk] = ppl
        ff = ppl / initial_ppls[pk] if initial_ppls[pk] > 0 else 0
        print(f"  {pk}: PPL={ppl:.2f}  (initial={initial_ppls[pk]:.2f}, FF={ff:.2f}x)")

    # ─── ANALYSIS ───
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print(f"\n  {'Domain':<8} {'Initial':<10} {'w/ LoRA':<10} {'No LoRA':<10} {'LoRA delta':<12} {'Base FF':<10} {'Verdict'}")
    print("  " + "-" * 80)

    for pk in phase_keys:
        init = initial_ppls[pk]
        w = with_lora[pk]
        wo = without_lora[pk]
        lora_delta = wo - w  # positive = LoRA was hurting this domain
        base_ff = wo / init if init > 0 else 0
        if abs(lora_delta) < 1.0:
            verdict = "LoRA neutral"
        elif lora_delta > 0:
            verdict = "LoRA HURT -> base worse w/o it"
        else:
            verdict = "LoRA HELPED -> base is weaker"
        print(f"  {pk:<8} {init:<10.2f} {w:<10.2f} {wo:<10.2f} {lora_delta:+.2f}       {base_ff:.2f}x     {verdict}")

    print("\n--- KEY QUESTION: Where is the forgetting? ---")
    a_base_ff = without_lora.get("A", 0) / initial_ppls.get("A", 1)
    a_lora_interference = with_lora.get("A", 0) - without_lora.get("A", 0)

    if a_base_ff < 1.15:
        print(f"\n  >> Base weights preserved A well (FF={a_base_ff:.2f}x)")
        print(f"     Forgetting is dominated by LoRA interference (delta={a_lora_interference:+.2f})")
        print(f"     V7 FIX: At inference, zero LoRA or use domain-specific LoRA routing")
        print(f"     This is GREAT news -- the accumulate approach works, we just need")
        print(f"     to handle the final LoRA at inference time!")
    elif a_base_ff < 1.4:
        print(f"\n  >> Base weights PARTIALLY preserved A (FF={a_base_ff:.2f}x)")
        print(f"     BOTH sources contribute:")
        print(f"       - Base weight corruption: {(a_base_ff - 1.0) * 100:.0f}% forgetting")
        print(f"       - LoRA interference: {a_lora_interference:+.2f} PPL points")
        print(f"     V7 FIX: Reduce merge interference + handle LoRA at inference")
    else:
        print(f"\n  >> Base weights LOST domain A (FF={a_base_ff:.2f}x)")
        print(f"     Merge itself causes corruption -- accumulate doesn't work as hoped")
        print(f"     V7 FIX: Need orthogonal constraints, EWC, or different approach")

    # Save results
    results = {
        "initial_ppls": initial_ppls,
        "with_active_lora": with_lora,
        "without_lora": without_lora,
        "lora_interference": {pk: without_lora[pk] - with_lora[pk] for pk in phase_keys},
        "base_forgetting_factor": {pk: without_lora[pk] / initial_ppls[pk] for pk in phase_keys},
        "total_forgetting_factor": {pk: with_lora[pk] / initial_ppls[pk] for pk in phase_keys},
    }
    os.makedirs("v6_results", exist_ok=True)
    with open("v6_results/diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: v6_results/diagnostic.json")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


run_diagnostic()
