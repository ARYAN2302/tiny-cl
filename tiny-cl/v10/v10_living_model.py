"""
V10: Living Model — SLAO + SVD Digest

After SLAO merge, the LoRA product has redundant directions.
SVD digest recompresses into optimal rank-r factors, keeping
only the highest-energy directions. The model "digests" after
each meal — absorbs what matters, drops what doesn't.

1. Train → SLAO merge (A=replace, B=interpolate)
2. SVD: B@A = U@S@V^T → keep top-r → new B, new A
3. A now has orthogonal rows (free ortho init for next task!)

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
    results_dir: str = "v10_results"

MODEL_CONFIG = ModelConfig()
DOMAINS = {
    "A": DomainConfig("medical", "Medical", "epfl-llm/guidelines", "clean_text"),
    "B": DomainConfig("code", "Code", "iamtarun/python_code_instructions_18k_alpaca", "output"),
    "C": DomainConfig("creative", "Creative", "roneneldan/TinyStories", "text"),
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
# SLAO + SVD DIGEST
# ──────────────────────────────────────────────

def get_lora_modules(model):
    from peft.tuners.lora.layer import LoraLayer
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, LoraLayer) and "default" in module.lora_A]

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state, device):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device))

def extract_orthogonal_A(model):
    ortho_A = {}
    for name, module in get_lora_modules(model):
        A = module.lora_A["default"].weight.data.float()
        Q, R = torch.linalg.qr(A.T.contiguous())
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        ortho_A[name] = Q.T
    print(f"  [ORTHO-A] QR from {len(ortho_A)} layers")
    return ortho_A

def initialize_slao(model, ortho_A, prev_ft_state, device):
    for name, module in get_lora_modules(model):
        if name in ortho_A:
            A_init = ortho_A[name].to(device)
            module.lora_A["default"].weight.data.copy_(A_init.to(module.lora_A["default"].weight.data.dtype))
        B_key = f"{name}.lora_B.default.weight"
        if prev_ft_state and B_key in prev_ft_state:
            B_val = prev_ft_state[B_key].to(device)
            module.lora_B["default"].weight.data.copy_(B_val.to(module.lora_B["default"].weight.data.dtype))
    print(f"  [SLAO-INIT] {len(get_lora_modules(model))} layers: A=ortho, B=prev_ft")

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

def svd_digest(model, device):
    """SVD digest: recompress B@A into optimal rank-r factors.
    
    delta = B @ A  (all accumulated knowledge)
    SVD: delta = U @ diag(S) @ V^T
    Keep top-r: new_B = U[:,:r]*sqrt(S[:r]), new_A = sqrt(S[:r])*V^T[:r,:]
    
    After digest, A has orthogonal rows → next task init comes for free.
    """
    total_energy = 0.0
    kept_energy = 0.0
    n_digested = 0
    sing_vals_all = []

    for name, module in get_lora_modules(model):
        A = module.lora_A["default"].weight.data.float()
        B = module.lora_B["default"].weight.data.float()
        delta = B @ A

        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = A.shape[0]

        total_e = (S ** 2).sum().item()
        kept_e = (S[:r] ** 2).sum().item()
        total_energy += total_e
        kept_energy += kept_e
        sing_vals_all.append(S[:min(r+4, len(S))].tolist())

        sqrt_S = torch.sqrt(S[:r])
        new_B = U[:, :r] * sqrt_S.unsqueeze(0)
        new_A = sqrt_S.unsqueeze(1) * Vh[:r, :]

        orig_dtype = module.lora_A["default"].weight.data.dtype
        module.lora_A["default"].weight.data.copy_(new_A.to(orig_dtype))
        module.lora_B["default"].weight.data.copy_(new_B.to(orig_dtype))
        n_digested += 1

    pct = 100 * kept_energy / max(total_energy, 1e-10)
    print(f"  [SVD-DIGEST] {n_digested} layers | energy: {pct:.1f}%")
    # Show singular value spectrum for first 3 modules
    for i, sv in enumerate(sing_vals_all[:3]):
        short = [f"{v:.2f}" for v in sv[:8]]
        print(f"    Layer {i}: S=[{', '.join(short)}{'...' if len(sv)>8 else ''}]")
    return pct


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
# RUN
# ──────────────────────────────────────────────

def run(seed=42):
    torch.manual_seed(seed); random.seed(seed)
    tc = TrainConfig(); mcfg = MODEL_CONFIG; device = tc.device

    print(f"\n{'#'*70}")
    print(f"# V10: LIVING MODEL — SLAO + SVD DIGEST")
    print(f"# Merge → SVD recompress → digest → repeat")
    print(f"# device={device}")
    print(f"{'#'*70}")

    model, tokenizer = create_model_conv_only(mcfg, device)
    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    merged_state = None
    prev_ft_state = None
    ortho_A = None
    completed = []
    task_num = 0
    results = {"method": "slao_svd_digest", "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        task_num += 1

        # Init
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if task_num == 1:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — standard fine-tune")
        else:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — SLAO init")
            ortho_A = extract_orthogonal_A(model)
            initialize_slao(model, ortho_A, prev_ft_state, device)

        # Train
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(trainable, lr=tc.lr, weight_decay=tc.weight_decay)
        loader = DataLoader(phases_data[pk], batch_size=mcfg.batch_size, shuffle=True, drop_last=False)
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
        prev_ft_state = get_lora_state(model)

        # SLAO merge
        if merged_state is None:
            merged_state = prev_ft_state.copy()
            print(f"  Task {task_num}: merged = finetuned")
        else:
            merged_state = slao_merge_B(merged_state, prev_ft_state, task_num, device)

        # Load merged state
        set_lora_state(model, merged_state, device)

        # SVD DIGEST — the key new step
        print(f"  [DIGEST] Recompressing...")
        energy_pct = svd_digest(model, device)

        # Save digested state as new merged
        merged_state = get_lora_state(model)

        # Eval
        ppls = {}
        for eval_pk in DOMAINS.keys():
            if val_ds[eval_pk] is not None:
                ppls[eval_pk] = compute_ppl(model, val_ds[eval_pk], device, tc.eval_samples)
        completed.append(pk)

        ev = {"perplexity": ppls, "avg_loss": tl/max(gs,1), "time": time.time()-t0, "svd_energy_pct": energy_pct}
        results["phases"][pk] = ev

        print(f"  Eval (digested LoRA):")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # Summary
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, "slao_svd.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"V10: LIVING MODEL — SLAO + SVD DIGEST")
    print(f"{'='*80}")
    keys = list(results["phases"].keys())
    if len(keys) >= 3:
        fp = results["phases"][keys[-1]].get("perplexity",{})
        a, b, c = fp.get("A",0), fp.get("B",0), fp.get("C",0)
        ff_a = a / results["phases"]["A"].get("perplexity",{}).get("A",1) if a > 0 else 0
        ff_b = b / results["phases"]["B"].get("perplexity",{}).get("B",1) if b > 0 else 0
        print(f"  A: {a:.1f}  B: {b:.1f}  C: {c:.1f}")
        print(f"  FF(A) = {ff_a:.2f}x   FF(B) = {ff_b:.2f}x")
        print()
        print(f"  Journey:")
        print(f"    naive:             FF(A)=1.64x")
        print(f"    V6 LoRA->base:     FF(A)=1.52x")
        print(f"    V9 SLAO:           FF(A)=1.16x")
        print(f"    V10 SLAO+SVD:      FF(A)={ff_a:.2f}x")
        print(f"    O-LoRA (routing):  FF(A)=1.00x")
    print(f"{'='*80}")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results

run()
