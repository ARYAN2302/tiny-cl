"""
V9: SLAO — CORRECT Implementation from ICLR 2026 Paper

Previous versions had the merge WRONG:
- V8: averaged both A and B → cross-term interference → FF=2.18x (worse than naive)
- V6: merged into base weights → corruption → FF=1.52x

This is the CORRECT Algorithm 1 from the paper:
1. Task 1: standard fine-tune → A_merge=A_1, B_merge=B_1
2. Task i (i>1):
   a. A_init = QR(prev_A)^T (orthogonal rows), B_init = prev_B (NOT zero!)
   b. Fine-tune both A and B on new task
   c. A_merge = A_ft (just replace), B_merge = B_merge + λ(B_ft - B_merge)
      where λ = 1/sqrt(i)

Living model: constant memory, no task routing, one LoRA knows everything.

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
    results_dir: str = "v9_results"

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
# SLAO CORE — CORRECT Algorithm 1
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
        # Sign correction
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        A_orth = Q.T  # [rank, in_features]
        ortho_A[name] = A_orth
    print(f"  [ORTHO-A] Extracted from {len(ortho_A)} LoRA layers (QR with sign correction)")
    return ortho_A


def get_lora_state(model, device="cpu"):
    """Save all LoRA param state."""
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}


def set_lora_state(model, state, device):
    """Load LoRA param state into model."""
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device))


def initialize_slao(model, ortho_A, prev_ft_B, device):
    """Algorithm 1 line 4:
    A_ft_i^(0) = Q_i^T  (orthogonal from QR of prev A)
    B_ft_i^(0) = B_ft_{i-1}  (previous task's fine-tuned B, NOT zero!)
    """
    from peft.tuners.lora.layer import LoraLayer
    n_init = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue

        # Set A from orthogonal basis
        if name in ortho_A:
            A_init = ortho_A[name].to(device)
            target_dtype = module.lora_A["default"].weight.data.dtype
            if A_init.dtype != target_dtype:
                A_init = A_init.to(target_dtype)
            module.lora_A["default"].weight.data.copy_(A_init)

        # Set B from previous fine-tuned B
        B_key = f"{name}.lora_B.default.weight"
        if B_key in prev_ft_B:
            B_val = prev_ft_B[B_key].to(device)
            target_dtype = module.lora_B["default"].weight.data.dtype
            if B_val.dtype != target_dtype:
                B_val = B_val.to(target_dtype)
            module.lora_B["default"].weight.data.copy_(B_val)

        n_init += 1
    print(f"  [SLAO-INIT] {n_init} layers: A=ortho(QR), B=prev_finetuned")


def slao_merge_B(merged_state, ft_state, task_num, device):
    """Algorithm 1 lines 6-7:
    A_merge = A_ft (just replace)
    B_merge = B_merge + λ(B_ft - B_merge), λ = 1/sqrt(i)
    """
    lam = 1.0 / math.sqrt(task_num)
    new_merged = {}

    for key in ft_state:
        ft_val = ft_state[key].to(device)

        if key in merged_state:
            if "lora_A" in key:
                # A: just replace with new fine-tuned A
                new_merged[key] = ft_val.cpu().clone()
            elif "lora_B" in key:
                # B: interpolate
                old_val = merged_state[key].to(device)
                merged_B = old_val + lam * (ft_val - old_val)
                new_merged[key] = merged_B.cpu().clone()
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
# MAIN
# ──────────────────────────────────────────────

def run(seed=42):
    torch.manual_seed(seed); random.seed(seed)
    tc = TrainConfig(); mcfg = MODEL_CONFIG; device = tc.device

    print(f"\n{'#'*70}")
    print(f"# V9: SLAO CORRECT (ICLR 2026 Algorithm 1)")
    print(f"# A=replace, B=interpolate(λ=1/√i), B_init=prev_ft")
    print(f"# device={device}")
    print(f"{'#'*70}")

    model, tokenizer = create_model_conv_only(mcfg, device)

    phases_data, val_ds = {}, {}
    for pk, d in DOMAINS.items():
        t, v = prepare_domain(d, tokenizer, mcfg.context_length, d.max_tokens, seed)
        phases_data[pk] = t; val_ds[pk] = v

    # SLAO state
    merged_state = None       # The living model's LoRA
    prev_ft_state = None      # Previous task's fine-tuned LoRA (for init)
    ortho_A = None            # Orthogonal basis from QR of prev A
    completed = []
    task_num = 0

    results = {"method": "slao_correct", "seed": seed, "phases": {}}

    for pk in DOMAINS.keys():
        task_num += 1
        dataset = phases_data[pk]

        # ── Initialize ──
        for n, p in model.named_parameters():
            if "lora_" in n: p.requires_grad = True

        if task_num == 1:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — standard fine-tune")
        else:
            print(f"\n  Phase {pk}: {DOMAINS[pk].display_name} — SLAO init (A=ortho, B=prev_ft)")
            ortho_A = extract_orthogonal_A(model)
            # Extract prev B from prev_ft_state
            prev_ft_B = {k: v for k, v in prev_ft_state.items() if "lora_B" in k}
            initialize_slao(model, ortho_A, prev_ft_B, device)

        # ── Train ──
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

        # ── Save fine-tuned state ──
        prev_ft_state = get_lora_state(model, device)

        # ── Merge ──
        if merged_state is None:
            merged_state = prev_ft_state.copy()
            print(f"  Task {task_num}: merged = finetuned (first task)")
        else:
            merged_state = slao_merge_B(merged_state, prev_ft_state, task_num, device)

        # ── Load merged LoRA for eval ──
        set_lora_state(model, merged_state, device)

        # ── Eval ──
        ppls = {}
        for eval_pk in DOMAINS.keys():
            if val_ds[eval_pk] is None: continue
            ppls[eval_pk] = compute_ppl(model, val_ds[eval_pk], device, tc.eval_samples)
        completed.append(pk)

        ev = {"perplexity": ppls, "avg_loss": tl / max(gs, 1), "time": time.time() - t0}
        results["phases"][pk] = ev

        print(f"  Eval (merged LoRA):")
        for p, ppl in ppls.items():
            print(f"    {p}: PPL={ppl:.2f}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, "slao_correct.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"V9: SLAO CORRECT — LIVING MODEL")
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
        print(f"  Comparison:")
        print(f"    naive (no CL):        FF(A)=1.64x")
        print(f"    V6 accumulate (base):  FF(A)=1.52x")
        print(f"    V8 slao_simple (wrong):FF(A)=1.26x")
        print(f"    V8 O-LoRA (separate):  FF(A)=1.00x (but needs task routing)")
        print(f"    V9 SLAO (this run):    FF(A)={ff_a:.2f}x (living model, no routing)")
        print()
        if ff_a < 1.1:
            print(f"  >> LIVING MODEL ACHIEVED!")
        elif ff_a < 1.3:
            print(f"  >> Close. Merge works but needs tuning.")
        else:
            print(f"  >> Still forgetting. The merge formula may need adjustment for LFM2.5.")
    print(f"{'='*80}")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return results


run()
