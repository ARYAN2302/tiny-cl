"""
avr-cl validation on LFM2.5-230M — INTENT CLASSIFICATION STREAM.

Plays to Liquid's strengths (IFEval 71.71, BFCLv3 43.26 — best-in-class at sub-300M).
Drops math (Liquid explicitly says "not recommended for reasoning-heavy workloads").

Stream: Banking77 (77 intents) → CLINC150 (150 intents) → SNIPS (7 intents) → Emotion (6 classes)
Model: LFM2.5-230M (hybrid: 8 conv + 6 attention blocks)
LoRA: r=128 on q_proj, k_proj, v_proj, o_proj, conv1d — targets BOTH conv + attention blocks

Conditions: Naive / AVR / Two-Stream+AVR (all three — this is the full validation)

T4 budget: ~5-6 hours total
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45.0", "peft>=0.13.0", "datasets>=3.0.0",
    "accelerate>=1.0.0", "numpy<2", "matplotlib", "sentencepiece",
    "protobuf", "scipy", "packaging"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

import json, time, random, math, gc, re, copy
from pathlib import Path
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "LiquidAI/LFM2.5-230M"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
# LFM2 hybrid: conv1d (conv blocks) + q/k/v/o_proj (attention blocks)
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "conv1d"]

TRAIN_LR = 2e-4
CONSOLIDATION_LR = 1e-4   # half of train LR (v34 setting)
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 3
CONSOLIDATION_EPOCHS = 1
BATCH_SIZE = 8
CONTEXT_LENGTH = 512
GRAD_ACCUM_STEPS = 2

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

BENCH_MAX_NEW_TOKENS = 50   # intent labels are short
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T1_MIN_ACCURACY = 0.40   # intent classification — Liquid's strength, should clear easily

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

_TOKENIZER = None
def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _TOKENIZER

def format_prompt(question: str) -> str:
    tok = _get_tokenizer()
    messages = [{"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def format_example(question: str, answer: str) -> str:
    tok = _get_tokenizer()
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return text + tok.eos_token

# ============================================================================
# INTENT CLASSIFICATION DATA
# ============================================================================
INTENT_TASKS = ["banking77", "clinc150", "snips", "emotion"]

def fmt_intent(text, label, label_names):
    """Format: 'Classify the intent: <text>\n\nIntent:' -> '<label_name>'"""
    prompt_q = f"Classify the intent of this text. Respond with only the intent label.\n\nText: {text}\n\nIntent:"
    gold = label_names[label] if isinstance(label, int) and label < len(label_names) else str(label)
    return (prompt_q, gold, gold)  # 3-tuple: prompt, train_answer, gold

def load_intent_stream():
    from datasets import load_dataset
    train_data, test_data = {}, {}
    rng = random.Random(SEED)

    print("    Loading Banking77...", flush=True)
    ds_tr = load_dataset("banking77", split="train")
    ds_te = load_dataset("banking77", split="test")
    label_names = ds_tr.features["label"].names
    tr = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_te]; rng.shuffle(te)
    train_data["banking77"] = tr[:2000]; test_data["banking77"] = te[:200]
    print(f"    banking77: {len(train_data['banking77'])} train, {len(test_data['banking77'])} test ({len(label_names)} intents)", flush=True)

    print("    Loading CLINC150...", flush=True)
    ds_tr = load_dataset("clinc_oos", "plus", split="train")
    ds_te = load_dataset("clinc_oos", "plus", split="test")
    n_intents = max(max(ex["intent"] for ex in ds_tr), max(ex["intent"] for ex in ds_te)) + 1
    label_names = [f"intent_{i}" for i in range(n_intents)]
    tr = [fmt_intent(ex["text"], ex["intent"], label_names) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_intent(ex["text"], ex["intent"], label_names) for ex in ds_te]; rng.shuffle(te)
    train_data["clinc150"] = tr[:2000]; test_data["clinc150"] = te[:200]
    print(f"    clinc150: {len(train_data['clinc150'])} train, {len(test_data['clinc150'])} test ({n_intents} intents)", flush=True)

    print("    Loading SNIPS...", flush=True)
    ds_tr = load_dataset("snips_built_in_intents", split="train")
    ds_te = load_dataset("snips_built_in_intents", split="test")
    label_names = ds_tr.features["label"].names
    tr = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_te]; rng.shuffle(te)
    train_data["snips"] = tr[:2000]; test_data["snips"] = te[:200]
    print(f"    snips: {len(train_data['snips'])} train, {len(test_data['snips'])} test ({len(label_names)} intents)", flush=True)

    print("    Loading Emotion...", flush=True)
    ds_tr = load_dataset("dair-ai/emotion", split="train")
    ds_te = load_dataset("dair-ai/emotion", split="test")
    label_names = ds_tr.features["label"].names
    tr = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_intent(ex["text"], ex["label"], label_names) for ex in ds_te]; rng.shuffle(te)
    train_data["emotion"] = tr[:2000]; test_data["emotion"] = te[:200]
    print(f"    emotion: {len(train_data['emotion'])} train, {len(test_data['emotion'])} test ({len(label_names)} classes)", flush=True)

    return train_data, test_data, INTENT_TASKS

# ============================================================================
# MODEL + LoRA
# ============================================================================
def create_model():
    global _TOKENIZER
    print(f"  Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _TOKENIZER = tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager")
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    print(f"  LoRA targets: {LORA_TARGETS}", flush=True)
    return model, tokenizer

# ============================================================================
# LoRA STATE + AVR (identical to v23/v34)
# ============================================================================
def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

def reset_lora_to_peft_init(model):
    """Reset LoRA to fresh PEFT init: lora_A = Kaiming uniform, lora_B = zeros.
    Used by two-stream hippocampus reset (v34)."""
    import torch.nn.init as init
    for n, p in model.named_parameters():
        if "lora_A" in n:
            init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()

def compute_ppl(model, tokenizer, examples, max_samples=100):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for prompt_q, train_answer, gold in examples[:max_samples]:
        text = format_example(prompt_q, train_answer)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))

def eval_all_ppls(model, tokenizer, train_data, task_order, trained_so_far, max_samples=100):
    ppls = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far: break
        ppls[task] = compute_ppl(model, tokenizer, train_data[task], max_samples)
    return ppls

def verify_drift(current_ppls, best_ppls, completed_tasks, threshold=DRIFT_THRESHOLD):
    drifted = {}
    for task in completed_tasks:
        if task not in current_ppls or task not in best_ppls: continue
        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0
        if ratio > threshold:
            drifted[task] = {"current_ppl": current_ppls[task], "best_ppl": best_ppls[task], "ratio": ratio}
    return drifted

def repair_toward_snapshot(model, snapshot_state, alpha=REPAIR_ALPHA):
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(DEVICE)
            p.data.copy_((1.0 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj

# ============================================================================
# TRAINING (LEARN phase + two-stream consolidation)
# ============================================================================
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

def build_training_stream(tokenizer, examples):
    all_tokens = []
    for prompt_q, train_answer, gold in examples:
        text = format_example(prompt_q, train_answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

def train_lora(model, tokenizer, examples, lr=TRAIN_LR, epochs=TASK_EPOCHS, tag="sft"):
    token_ids = build_training_stream(tokenizer, examples)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=TRAIN_WD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0; opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step(); opt.zero_grad(); accum = 0
                gs += 1; tl += out.loss.item()
                if gs % 50 == 0:
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)
    return gs, tl

def consolidate_to_neocortex(model, tokenizer, hippo_state, neo_state, examples, epochs=CONSOLIDATION_EPOCHS):
    """KL distill hippocampus -> neocortex (v34 two-stream consolidation)."""
    print(f"    [consolid] KL distill hippocampus -> neocortex ({epochs} epoch)", flush=True)
    token_ids = build_training_stream(tokenizer, examples)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=CONSOLIDATION_LR, weight_decay=TRAIN_WD)
    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0; opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            # 1. Hippocampus logits (no grad)
            set_lora_state(model, hippo_state)
            model.eval()
            with torch.no_grad():
                hippo_logits = model(input_ids=input_ids).logits
            # 2. Neocortex logits (with grad)
            set_lora_state(model, neo_state)
            model.train()
            neo_logits = model(input_ids=input_ids).logits
            # 3. KL(p_hippo || p_neo)
            shift_hippo = hippo_logits[..., :-1, :].contiguous()
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            p_hippo = F.softmax(shift_hippo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            (kl_loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step(); opt.zero_grad(); accum = 0
                gs += 1; tl += kl_loss.item()
                if gs % 50 == 0:
                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state

# ============================================================================
# EVALUATION (batched)
# ============================================================================
def generate_batch(model, tokenizer, prompt_qs, max_new_tokens=BENCH_MAX_NEW_TOKENS, batch_size=16):
    results = []
    gc_was = getattr(model, "gradient_checkpointing", False)
    if gc_was:
        try: model.gradient_checkpointing_disable()
        except: pass
    model.eval()
    try:
        for i in range(0, len(prompt_qs), batch_size):
            batch = prompt_qs[i:i+batch_size]
            texts = [format_prompt(q) for q in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=1.0)
            for out in outputs:
                input_len = inputs["input_ids"].shape[1]
                results.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())
    finally:
        if gc_was:
            try:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            except: pass
    return results

def normalize_answer(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s.-]', ' ', s)
    return ' '.join(s.split())

def score(response, gold):
    """Intent classification: normalized text match."""
    return 1.0 if normalize_answer(response) == normalize_answer(gold) else 0.0

def evaluate_task(model, tokenizer, test_examples, task_name, max_questions=200, batch_size=16):
    print(f"    Eval {task_name} ({min(len(test_examples), max_questions)} Qs, batch={batch_size})...", flush=True)
    total = min(len(test_examples), max_questions)
    examples = test_examples[:total]
    prompt_qs = [ex[0] for ex in examples]
    golds = [ex[2] for ex in examples]
    t0 = time.time()
    correct = 0; processed = 0
    for bs in range(0, len(prompt_qs), batch_size):
        be = min(bs + batch_size, len(prompt_qs))
        bp = prompt_qs[bs:be]; bg = golds[bs:be]
        responses = generate_batch(model, tokenizer, bp, max_new_tokens=BENCH_MAX_NEW_TOKENS, batch_size=len(bp))
        for r, g in zip(responses, bg):
            if score(r, g): correct += 1
            processed += 1
        if processed % (batch_size * 2) < batch_size or processed >= total:
            print(f"      [{processed}/{total}] acc={correct/processed:.3f} ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc

def compute_metrics(R, task_order):
    T = len(task_order)
    ACC = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt = [R[T-1][j] - R[j][j] for j in range(T-1)]
    BWT = float(np.mean(bwt)) if bwt else 0.0
    ff = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]
    FF = float(np.mean(ff)) if ff else 0.0
    return {"ACC": ACC, "BWT": BWT, "FF": FF}

# ============================================================================
# CHECKPOINT
# ============================================================================
def save_partial(condition, results, task_idx):
    path = OUTPUT_DIR / f"partial_intent_{condition}_task{task_idx+1}.json"
    with open(path, "w") as f:
        json.dump({"condition": condition, "task": task_idx+1, "results": results}, f, indent=2, default=str)
    print(f"  [checkpoint] saved {path.name}", flush=True)

# ============================================================================
# CONDITION 1: NAIVE
# ============================================================================
def run_naive(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 1: Naive SFT\n{'#'*70}", flush=True)
    model, tokenizer = create_model()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task} (naive)\n{'='*60}", flush=True)
        train_lora(model, tokenizer, train_data[task], tag="naive")
        print(f"\n  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate_task(model, tokenizer, test_data[task_order[j]], task_order[j])
        if ti == 0:
            print(f"\n  [SANITY GATE] T1 ({task}) = {R[0][0]:.3f} (need >= {T1_MIN_ACCURACY})", flush=True)
            if R[0][0] < T1_MIN_ACCURACY:
                msg = f"ABORT: T1 {R[0][0]:.3f} < {T1_MIN_ACCURACY}"
                print(f"  {msg}", flush=True)
                with open(OUTPUT_DIR / "RUN_FAILED.txt", "w") as f: f.write(f"{msg}\n")
                raise RuntimeError(msg)
            print(f"  [SANITY GATE] PASSED", flush=True)
        save_partial("naive", {"R": R, "metrics": compute_metrics(R, task_order)}, ti)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [naive] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f}", flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics

# ============================================================================
# CONDITION 2: AVR (v23 style)
# ============================================================================
def run_avr(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 2: AVR (SFT + PPL verify + closed-form repair)\n{'#'*70}", flush=True)
    model, tokenizer = create_model()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    snapshot = None; best_ppls = {}; completed = []
    total_repairs = 0; log = []
    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task} (AVR)\n{'='*60}", flush=True)
        train_lora(model, tokenizer, train_data[task], tag="avr")
        post_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
        if task not in best_ppls: best_ppls[task] = post_ppls[task]
        print(f"  Post-train PPLs: " + " | ".join(f"{k}:{v:.2f}" for k,v in post_ppls.items()), flush=True)
        repairs = 0
        if ti > 0 and snapshot is not None:
            drifted = verify_drift(post_ppls, best_ppls, completed)
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f}/{info['best_ppl']:.2f}={info['ratio']:.2f}x", flush=True)
                still = drifted
                for step in range(MAX_REPAIR_STEPS):
                    n_adj = repair_toward_snapshot(model, snapshot)
                    repairs += 1
                    rp = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
                    still = verify_drift(rp, best_ppls, completed)
                    print(f"    [AVR] Repair {step+1}: {n_adj} params, drifted: {list(still.keys()) if still else 'none'}", flush=True)
                    if not still:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still: print(f"  [AVR] Max steps reached, drift remains", flush=True)
            else: print(f"  [AVR] No drift", flush=True)
        total_repairs += repairs
        log.append({"task": task, "repair_steps": repairs})
        fp = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
        for dk, dp in fp.items():
            if dk not in best_ppls or dp < best_ppls[dk]: best_ppls[dk] = dp
        snapshot = get_lora_state(model)
        completed.append(task)
        print(f"\n  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate_task(model, tokenizer, test_data[task_order[j]], task_order[j])
        if ti == 0:
            print(f"\n  [SANITY GATE] T1 ({task}) = {R[0][0]:.3f} (need >= {T1_MIN_ACCURACY})", flush=True)
            if R[0][0] < T1_MIN_ACCURACY:
                msg = f"ABORT: T1 {R[0][0]:.3f} < {T1_MIN_ACCURACY}"
                print(f"  {msg}", flush=True)
                with open(OUTPUT_DIR / "RUN_FAILED.txt", "w") as f: f.write(f"{msg}\n")
                raise RuntimeError(msg)
            print(f"  [SANITY GATE] PASSED", flush=True)
        save_partial("avr", {"R": R, "metrics": compute_metrics(R, task_order), "total_repairs": total_repairs, "log": log}, ti)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [AVR] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f} repairs={total_repairs}", flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics, total_repairs, log

# ============================================================================
# CONDITION 3: TWO-STREAM + AVR (v34 condition C)
# ============================================================================
def run_twostream_avr(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 3: Two-Stream + AVR\n# (hippocampus + KL distill + AVR on neocortex)\n{'#'*70}", flush=True)
    model, tokenizer = create_model()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    neo_state = get_lora_state(model)
    best_ppls = {}; completed = []
    total_repairs = 0; log = []
    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task} (two-stream+AVR)\n{'='*60}", flush=True)
        neo_snapshot = copy.deepcopy(neo_state)
        print(f"  [twostream] Neocortex snapshot taken", flush=True)
        reset_lora_to_peft_init(model)
        print(f"  [twostream] Hippocampus reset to fresh PEFT init", flush=True)
        train_lora(model, tokenizer, train_data[task], lr=TRAIN_LR, tag="hippo")
        hippo_state = get_lora_state(model)
        print(f"  [twostream] Hippocampus trained", flush=True)
        set_lora_state(model, neo_state)
        neo_state = consolidate_to_neocortex(model, tokenizer, hippo_state, neo_state, train_data[task])
        print(f"  [twostream] Consolidation complete", flush=True)
        repairs = 0
        set_lora_state(model, neo_state)
        post_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
        if task not in best_ppls: best_ppls[task] = post_ppls[task]
        print(f"  Post-consolid PPLs: " + " | ".join(f"{k}:{v:.2f}" for k,v in post_ppls.items()), flush=True)
        if ti > 0:
            drifted = verify_drift(post_ppls, best_ppls, completed)
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f}/{info['best_ppl']:.2f}={info['ratio']:.2f}x", flush=True)
                still = drifted
                for step in range(MAX_REPAIR_STEPS):
                    n_adj = repair_toward_snapshot(model, neo_snapshot)
                    repairs += 1
                    rp = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
                    still = verify_drift(rp, best_ppls, completed)
                    print(f"    [AVR] Repair {step+1}: {n_adj} params, drifted: {list(still.keys()) if still else 'none'}", flush=True)
                    if not still:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still: print(f"  [AVR] Max steps reached, drift remains", flush=True)
                neo_state = get_lora_state(model)
            else: print(f"  [AVR] No drift", flush=True)
        total_repairs += repairs
        log.append({"task": task, "repair_steps": repairs})
        fp = eval_all_ppls(model, tokenizer, train_data, task_order, ti+1)
        for dk, dp in fp.items():
            if dk not in best_ppls or dp < best_ppls[dk]: best_ppls[dk] = dp
        completed.append(task)
        print(f"\n  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate_task(model, tokenizer, test_data[task_order[j]], task_order[j])
        if ti == 0:
            print(f"\n  [SANITY GATE] T1 ({task}) = {R[0][0]:.3f} (need >= {T1_MIN_ACCURACY})", flush=True)
            if R[0][0] < T1_MIN_ACCURACY:
                msg = f"ABORT: T1 {R[0][0]:.3f} < {T1_MIN_ACCURACY}"
                print(f"  {msg}", flush=True)
                with open(OUTPUT_DIR / "RUN_FAILED.txt", "w") as f: f.write(f"{msg}\n")
                raise RuntimeError(msg)
            print(f"  [SANITY GATE] PASSED", flush=True)
        save_partial("twostream", {"R": R, "metrics": compute_metrics(R, task_order), "total_repairs": total_repairs, "log": log}, ti)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [twostream+AVR] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f} repairs={total_repairs}", flush=True)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics, total_repairs, log

# ============================================================================
# HEATMAP
# ============================================================================
def make_heatmap(results, task_order, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Naive SFT", "AVR (verify + repair)", "Two-Stream + AVR"]
    keys = ["naive", "avr", "twostream"]
    short = [t[:12] for t in task_order]
    for ax, title, key in zip(axes, titles, keys):
        if key not in results: continue
        R = np.array(results[key]["R"])
        im = ax.imshow(R, cmap='RdYlGn', vmin=0, vmax=0.8, aspect='auto')
        ax.set_xticks(range(len(short))); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(short))); ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Eval task"); ax.set_ylabel("After training task")
        m = results[key]["metrics"]
        extra = f"  repairs={results[key].get('total_repairs',0)}" if key != "naive" else ""
        ax.set_title(f"{title}\nACC={m['ACC']:.3f}  BWT={m['BWT']:+.3f}  FF={m['FF']:.3f}{extra}", fontsize=10)
        for i in range(len(task_order)):
            for j in range(len(task_order)):
                if j <= i:
                    ax.text(j, i, f"{R[i,j]:.2f}", ha='center', va='center', fontsize=9,
                            color='black' if R[i,j] > 0.4 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"avr-cl [intent stream] — {MODEL_ID}, LoRA r={LORA_RANK} (conv+attn), seed {SEED}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved: {save_path}", flush=True)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("avr-cl validation: LFM2.5-230M (hybrid conv+attention)", flush=True)
print(f"Model: {MODEL_ID} | Seed: {SEED}", flush=True)
print(f"Stream: Intent classification (Banking77 → CLINC150 → SNIPS → Emotion)", flush=True)
print(f"LoRA: rank={LORA_RANK}, targets={LORA_TARGETS} (BOTH conv + attention)", flush=True)
print(f"Conditions: Naive / AVR / Two-Stream+AVR (all three)", flush=True)
print(f"AVR: threshold={DRIFT_THRESHOLD}, alpha={REPAIR_ALPHA}, max_steps={MAX_REPAIR_STEPS}", flush=True)
print(f"Batch: {BATCH_SIZE} (accum {GRAD_ACCUM_STEPS} = {BATCH_SIZE*GRAD_ACCUM_STEPS} eff), ctx {CONTEXT_LENGTH}", flush=True)
print(f"Sanity gate: T1 (banking77) >= {T1_MIN_ACCURACY}", flush=True)
print("="*70, flush=True)

print(f"\nLoading intent stream data...", flush=True)
train_data, test_data, task_order = load_intent_stream()

results = {}

# Condition 1: Naive
R_n, m_n = run_naive(train_data, test_data, task_order)
results["naive"] = {"R": R_n, "metrics": m_n}
print(f"\n  Naive: ACC={m_n['ACC']:.3f} BWT={m_n['BWT']:+.3f} FF={m_n['FF']:.3f}", flush=True)

# Condition 2: AVR
R_a, m_a, repairs_a, log_a = run_avr(train_data, test_data, task_order)
results["avr"] = {"R": R_a, "metrics": m_a, "total_repairs": repairs_a, "log": log_a}
print(f"\n  AVR: ACC={m_a['ACC']:.3f} BWT={m_a['BWT']:+.3f} FF={m_a['FF']:.3f} repairs={repairs_a}", flush=True)

# Condition 3: Two-Stream + AVR
R_t, m_t, repairs_t, log_t = run_twostream_avr(train_data, test_data, task_order)
results["twostream"] = {"R": R_t, "metrics": m_t, "total_repairs": repairs_t, "log": log_t}
print(f"\n  TwoStream+AVR: ACC={m_t['ACC']:.3f} BWT={m_t['BWT']:+.3f} FF={m_t['FF']:.3f} repairs={repairs_t}", flush=True)

# Heatmap + JSON
make_heatmap(results, task_order, OUTPUT_DIR / "heatmap_intent_lfm230m.png")
with open(OUTPUT_DIR / "results_intent_lfm230m.json", "w") as f:
    json.dump({"stream": "intent", "model": MODEL_ID, "seed": SEED,
        "config": {"lora_rank": LORA_RANK, "lora_targets": LORA_TARGETS,
            "drift_threshold": DRIFT_THRESHOLD, "repair_alpha": REPAIR_ALPHA,
            "max_repair_steps": MAX_REPAIR_STEPS, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS, "context_length": CONTEXT_LENGTH,
            "tasks": task_order, "conditions": ["naive", "avr", "twostream"]},
        "results": results}, f, indent=2, default=str)
print(f"\nResults saved: results_intent_lfm230m.json", flush=True)

# Final summary
print(f"\n{'='*80}", flush=True)
print("FINAL SUMMARY [LFM2.5-230M — Intent Classification]", flush=True)
print(f"{'='*80}", flush=True)
print(f"\n{'Method':<30} {'ACC':<8} {'BWT':<8} {'FF':<8} {'Repairs':<10}", flush=True)
print("-"*70, flush=True)
print(f"{'Naive SFT':<30} {m_n['ACC']:<8.3f} {m_n['BWT']:<+8.3f} {m_n['FF']:<8.3f} {'-':<10}", flush=True)
print(f"{'AVR (v23 style)':<30} {m_a['ACC']:<8.3f} {m_a['BWT']:<+8.3f} {m_a['FF']:<8.3f} {repairs_a:<10}", flush=True)
print(f"{'Two-Stream + AVR (v34)':<30} {m_t['ACC']:<8.3f} {m_t['BWT']:<+8.3f} {m_t['FF']:<8.3f} {repairs_t:<10}", flush=True)
print("-"*70, flush=True)
print(f"\n  Recovery (AVR vs Naive):       BWT {m_n['BWT']:+.3f} -> {m_a['BWT']:+.3f} ({(m_a['BWT']-m_n['BWT'])*100:+.1f}pp)", flush=True)
print(f"  Recovery (TwoStream vs Naive): BWT {m_n['BWT']:+.3f} -> {m_t['BWT']:+.3f} ({(m_t['BWT']-m_n['BWT'])*100:+.1f}pp)", flush=True)

T = len(task_order)
t1n = R_n[T-1][0]; t1a = R_a[T-1][0]; t1t = R_t[T-1][0]
print(f"\n  Task 1 ({task_order[0]}) after all {T} tasks:", flush=True)
print(f"    Naive:          {t1n:.3f}", flush=True)
print(f"    AVR:            {t1a:.3f}  (recovered {(t1a-t1n)*100:+.1f}pp)", flush=True)
print(f"    Two-Stream+AVR: {t1t:.3f}  (recovered {(t1t-t1n)*100:+.1f}pp)", flush=True)

# R-matrices
for cond, R in [("naive", R_n), ("avr", R_a), ("twostream", R_t)]:
    print(f"\n  R MATRIX ({cond}):", flush=True)
    header = "After\\Test  " + "  ".join(f"{t[:12]:<14}" for t in task_order)
    print(f"  {header}", flush=True)
    for i in range(len(task_order)):
        row = f"  {task_order[i][:12]:<14} " + "  ".join(f"{R[i][j]:<14.3f}" for j in range(len(task_order)))
        print(row, flush=True)

summary = {"lfm2_230m_intent": {
    "naive": m_n,
    "avr": {**m_a, "total_repairs": repairs_a},
    "twostream": {**m_t, "total_repairs": repairs_t},
}}
with open(OUTPUT_DIR / "summary_intent_lfm230m.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSummary saved: summary_intent_lfm230m.json", flush=True)
print(f"\nDONE.", flush=True)
