"""
V8.1 DIAGNOSTIC: Why does O-LoRA have 1.23x forgetting?

If we load A's exact LoRA and base weights never changed, PPL should be 16.78.
But it's 20.6. Something is changing that we're not capturing.

This script trains just A+B (2 domains, faster), then:
1. Records PPL on A right after training A (gold standard)
2. Trains B with orthogonal init
3. Loads A's LoRA back, checks PPL again
4. Compares ALL model state (params + buffers) between step 1 and step 3
5. Identifies exactly what changed

USAGE: Copy-paste into one Kaggle cell. Run it. ~30 min.
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

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
}

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
    ds = load_dataset(path=domain.dataset_name, split=domain.split)
    texts = [t for t in ds[domain.text_field] if t and len(t.strip()) > 10]
    random.seed(seed); random.shuffle(texts)
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        if len(all_tokens) >= max_tokens: break
    token_ids = torch.tensor(all_tokens[:int(max_tokens)], dtype=torch.long)
    n_val = min(int(len(token_ids) * 0.1), 100_000)
    n_train = len(token_ids) - n_val
    return TextDataset(token_ids[:n_train], context_length), TextDataset(token_ids[n_train:n_train + n_val], context_length)

def is_conv_module(name):
    for idx in CONV_LAYER_IDS:
        if f"layers.{idx}." in name: return True
    return False

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

def snapshot_model_state(model):
    """Save EVERYTHING: all parameters AND buffers."""
    state = {}
    for name, param in model.named_parameters():
        state[name] = param.data.cpu().clone()
    for name, buf in model.named_buffers():
        state[name] = buf.data.cpu().clone()
    return state

def compare_states(state1, state2, label1="before", label2="after"):
    """Find all differences between two model states."""
    changes = []
    all_keys = set(list(state1.keys()) + list(state2.keys()))
    for key in sorted(all_keys):
        if key not in state1:
            changes.append({"key": key, "change": "NEW", "norm2": state2[key].norm().item()})
            continue
        if key not in state2:
            changes.append({"key": key, "change": "DELETED", "norm1": state1[key].norm().item()})
            continue
        diff = (state1[key].float() - state2[key].float()).norm().item()
        if diff > 1e-8:
            rel = diff / max(state1[key].float().norm().item(), 1e-8)
            changes.append({
                "key": key,
                "change": "MODIFIED",
                "abs_diff": diff,
                "rel_diff": rel,
                "norm_before": state1[key].float().norm().item(),
                "norm_after": state2[key].float().norm().item(),
            })
    return changes


def run_diagnostic():
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tc = TrainConfig(); mcfg = MODEL_CONFIG; device = tc.device
    torch.manual_seed(42); random.seed(42)

    print("=" * 70)
    print("V8.1 DIAGNOSTIC: Why does O-LoRA have 1.23x forgetting?")
    print("=" * 70)

    # Load model
    print(f"  Loading {mcfg.hf_id}...")
    tokenizer = AutoTokenizer.from_pretrained(mcfg.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mcfg.hf_id, trust_remote_code=True,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    lora_config = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=mcfg.conv_targets, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, 42)
        phases_data[pk] = t; val_ds[pk] = v

    # ─── STEP 1: Train A, record PPL, snapshot everything ───
    print("\n" + "=" * 70)
    print("STEP 1: Train domain A")
    print("=" * 70)

    for n, p in model.named_parameters():
        if "lora_" in n and is_conv_module(n): p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
    loader = DataLoader(phases_data["A"], batch_size=mcfg.batch_size, shuffle=True)
    gs, tl = 0, 0.0
    for epoch in range(tc.epochs_per_phase):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

    ppl_A_after_train = compute_ppl(model, val_ds["A"], device, tc.eval_samples)
    state_after_A = snapshot_model_state(model)
    # Also save just the LoRA params for O-LoRA restore
    lora_state_A = {n: p.data.cpu().clone() for n, p in model.named_parameters()
                    if "lora_" in n and is_conv_module(n)}
    print(f"\n  PPL on A after training A: {ppl_A_after_train:.2f}  ← GOLD STANDARD")
    print(f"  Saved full model state ({len(state_after_A)} entries)")

    # ─── STEP 2: Extract orthogonal basis, train B ───
    print("\n" + "=" * 70)
    print("STEP 2: Train domain B (with orthogonal init)")
    print("=" * 70)

    # Extract basis from A's LoRA
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

    # Re-init LoRA and project orthogonally
    for n, p in model.named_parameters():
        if "lora_A" in n and is_conv_module(n): nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        if "lora_B" in n and is_conv_module(n): p.data.zero_()

    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if not is_conv_module(name): continue
        if "default" not in module.lora_A: continue
        if name not in basis: continue
        A = module.lora_A["default"].weight.data
        B = module.lora_B["default"].weight.data
        V = basis[name]["A_right"].to(A.device)
        U = basis[name]["B_left"].to(B.device)
        proj_A = V.T @ V
        null_A = torch.eye(proj_A.shape[0], device=A.device) - proj_A
        A_orth = A @ null_A
        proj_B = U @ U.T
        null_B = torch.eye(proj_B.shape[0], device=B.device) - proj_B
        B_orth = null_B @ B
        if A_orth.norm() > 1e-8: A_orth = A_orth * (A.norm() / A_orth.norm())
        if B_orth.norm() > 1e-8: B_orth = B_orth * (B.norm() / B_orth.norm())
        module.lora_A["default"].weight.data.copy_(A_orth)
        module.lora_B["default"].weight.data.copy_(B_orth)

    print("  Orthogonal init done")

    # Train B
    for n, p in model.named_parameters():
        if "lora_" in n and is_conv_module(n): p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
    loader = DataLoader(phases_data["B"], batch_size=mcfg.batch_size, shuffle=True)
    gs, tl = 0, 0.0
    for epoch in range(tc.epochs_per_phase):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, tc.max_grad_norm)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 100 == 0: print(f"    step {gs} | avg_loss={tl/gs:.4f}")

    ppl_B_after_train = compute_ppl(model, val_ds["B"], device, tc.eval_samples)
    print(f"\n  PPL on B after training B: {ppl_B_after_train:.2f}")

    # ─── STEP 3: Load A's LoRA back, check PPL ───
    print("\n" + "=" * 70)
    print("STEP 3: Load A's exact LoRA back, measure PPL")
    print("=" * 70)

    # Load A's LoRA
    for n, p in model.named_parameters():
        if n in lora_state_A:
            p.data.copy_(lora_state_A[n].to(device))

    ppl_A_with_restored_lora = compute_ppl(model, val_ds["A"], device, tc.eval_samples)
    state_after_restore = snapshot_model_state(model)
    print(f"  PPL on A with restored LoRA: {ppl_A_with_restored_lora:.2f}")
    print(f"  Gold standard PPL on A:      {ppl_A_after_train:.2f}")
    ff = ppl_A_with_restored_lora / ppl_A_after_train
    print(f"  FF = {ff:.2f}x")

    # ─── STEP 4: Compare full model states ───
    print("\n" + "=" * 70)
    print("STEP 4: What changed between 'after A' and 'after restore'?")
    print("=" * 70)

    changes = compare_states(state_after_A, state_after_restore, "after_A_train", "after_restore")
    if not changes:
        print("  NOTHING changed! PPL difference must be numerical precision.")
    else:
        print(f"  {len(changes)} parameters/buffers changed:\n")
        # Categorize changes
        lora_changes = [c for c in changes if "lora_" in c["key"]]
        base_changes = [c for c in changes if "lora_" not in c["key"] and "base_layer" not in c["key"]]
        norm_changes = [c for c in changes if "norm" in c["key"].lower()]
        buffer_changes = [c for c in changes if c["change"] == "MODIFIED" and "norm" not in c["key"].lower()]

        if lora_changes:
            print(f"  LoRA changes ({len(lora_changes)}):")
            for c in lora_changes[:5]:
                if c["change"] == "MODIFIED":
                    print(f"    {c['key']}: rel_diff={c['rel_diff']:.6f} abs={c['abs_diff']:.6f}")
                else:
                    print(f"    {c['key']}: {c['change']}")
            if len(lora_changes) > 5:
                print(f"    ... and {len(lora_changes)-5} more")

        if norm_changes:
            print(f"\n  Norm changes ({len(norm_changes)}):")
            for c in norm_changes[:10]:
                if c["change"] == "MODIFIED":
                    print(f"    {c['key']}: rel_diff={c['rel_diff']:.6f}")
                else:
                    print(f"    {c['key']}: {c['change']}")

        if base_changes:
            print(f"\n  Other changes ({len(base_changes)}):")
            for c in base_changes[:10]:
                if c["change"] == "MODIFIED":
                    print(f"    {c['key']}: rel_diff={c['rel_diff']:.6f}")
                else:
                    print(f"    {c['key']}: {c['change']}")

    # ─── STEP 5: Full restore (everything, not just LoRA) ───
    print("\n" + "=" * 70)
    print("STEP 5: Full restore of ALL params + buffers")
    print("=" * 70)

    # Restore everything from state_after_A
    for name, param in model.named_parameters():
        if name in state_after_A:
            param.data.copy_(state_after_A[name].to(device))
    for name, buf in model.named_buffers():
        if name in state_after_A:
            buf.data.copy_(state_after_A[name].to(device))

    ppl_A_full_restore = compute_ppl(model, val_ds["A"], device, tc.eval_samples)
    print(f"  PPL on A with FULL restore: {ppl_A_full_restore:.2f}")
    print(f"  Gold standard PPL on A:     {ppl_A_after_train:.2f}")
    ff_full = ppl_A_full_restore / ppl_A_after_train
    print(f"  FF = {ff_full:.2f}x")

    # ─── SUMMARY ───
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PPL on A right after training:  {ppl_A_after_train:.2f}")
    print(f"  PPL on A with LoRA restore only: {ppl_A_with_restored_lora:.2f}  (FF={ff:.2f}x)")
    print(f"  PPL on A with FULL restore:      {ppl_A_full_restore:.2f}  (FF={ff_full:.2f}x)")
    print()
    if ff_full > 1.02:
        print("  >> Even FULL state restore doesn't match! This means float16 precision")
        print("     loss during save/load is causing the gap. Try float32.")
    elif ff < 1.02 and ff_full < 1.02:
        print("  >> Full restore works! The gap was from state we weren't saving.")
        print("     The O-LoRA implementation is missing some state in save/restore.")
    elif ff > 1.05 and ff_full < 1.02:
        print("  >> LoRA-only restore loses info, but full restore works.")
        print("     Something besides LoRA params changed (buffers, norm stats, etc.)")
    else:
        print("  >> Both restores lose info. Likely float16 precision issue.")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


run_diagnostic()
