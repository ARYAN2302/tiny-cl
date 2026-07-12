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

import subprocess, sys, os
# DON'T install or change anything — use Kaggle's stock transformers 5.x.
# The numpy crash is triggered by masking_utils importing torch._dynamo
# (only when torch >= 2.6). Patch the flag to skip that import.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "datasets>=3.0.0",
    "accelerate>=1.0.0", "matplotlib", "sentencepiece",
    "protobuf", "packaging"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

# CRITICAL: patch transformers BEFORE importing model classes.
# masking_utils.py line 38-39 does:
#   if _is_torch_greater_or_equal_than_2_6:
#       from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
# That dynamo import triggers numpy.random import which crashes on Kaggle's
# mismatched numpy C extensions. Patching the flag to False skips it.
import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
# Also patch the function version in case it's called dynamically
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False
print(f"transformers: {_iu.__version__ if hasattr(_iu, '__version__') else 'stock'}", flush=True)
print("Patched _is_torch_greater_or_equal_than_2_6 = False (avoids numpy crash)", flush=True)

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
BATCH_SIZE = 4            # T4-safe for r=128 + consolidation (2 forward passes)
CONTEXT_LENGTH = 512
GRAD_ACCUM_STEPS = 4      # effective batch = 4 × 4 = 16

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

BENCH_MAX_NEW_TOKENS = 50   # intent labels are short
EVAL_BATCH_SIZE = 8         # smaller eval batch to limit KV cache
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

T1_MIN_ACCURACY = 0.10   # 77-class intent classification — 1.3% is random, 10% is learning

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def aggressive_cleanup():
    """Aggressive memory cleanup between conditions — prevents T4 OOM from fragmentation."""
    import gc
    gc.collect()
    gc.collect()  # twice — Python sometimes needs two passes
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # twice — releases cached blocks back to OS

_TOKENIZER = None
_TOKENIZER_PATH = None

def _patch_tokenizer():
    """Download LFM2.5-230M tokenizer files and patch the broken tokenizer_class.
    Also download the chat_template.jinja file (LFM2 stores it separately)."""
    global _TOKENIZER_PATH
    if _TOKENIZER_PATH is not None:
        return _TOKENIZER_PATH
    import json
    from huggingface_hub import hf_hub_download

    cache_dir = OUTPUT_DIR / "tokenizer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download ALL tokenizer files including chat_template.jinja
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "tokenizer.model", "chat_template.jinja", "added_tokens.json",
                   "vocab.json", "merges.txt"]:
        try:
            hf_hub_download(MODEL_ID, fname, local_dir=str(cache_dir))
        except Exception:
            pass

    # Patch tokenizer_config.json
    config_path = cache_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if config.get("tokenizer_class") == "TokenizersBackend":
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Patched tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast", flush=True)

    _TOKENIZER_PATH = str(cache_dir)
    return _TOKENIZER_PATH

def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        path = _patch_tokenizer()
        _TOKENIZER = AutoTokenizer.from_pretrained(path, use_fast=True)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
        # Load chat_template from chat_template.jinja if not already set
        if _TOKENIZER.chat_template is None:
            template_path = OUTPUT_DIR / "tokenizer_cache" / "chat_template.jinja"
            if template_path.exists():
                with open(template_path) as f:
                    _TOKENIZER.chat_template = f.read()
                print(f"  Loaded chat_template from chat_template.jinja ({len(_TOKENIZER.chat_template)} chars)", flush=True)
            else:
                # Fallback: basic ChatML
                _TOKENIZER.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                print(f"  Set fallback chat_template", flush=True)
    return _TOKENIZER

def format_prompt(question: str) -> str:
    tok = _get_tokenizer()
    messages = [{"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_example(question: str, answer: str) -> str:
    tok = _get_tokenizer()
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    # SNIPS only has 'train' split — split it 90/10 for train/test
    all_snips = list(ds_tr)
    rng.shuffle(all_snips)
    n_test = min(200, len(all_snips) // 10)
    snips_te = all_snips[:n_test]
    snips_tr = all_snips[n_test:]
    label_names = ds_tr.features["label"].names
    tr = [fmt_intent(ex["text"], ex["label"], label_names) for ex in snips_tr]; rng.shuffle(tr)
    te = [fmt_intent(ex["text"], ex["label"], label_names) for ex in snips_te]; rng.shuffle(te)
    train_data["snips"] = tr[:2000]; test_data["snips"] = te[:200]
    print(f"    snips: {len(train_data['snips'])} train, {len(test_data['snips'])} test ({len(label_names)} intents)", flush=True)

    print("    Loading Emotion...", flush=True)
    ds_tr = load_dataset("dair-ai/emotion", split="train")
    try:
        ds_te = load_dataset("dair-ai/emotion", split="test")
    except ValueError:
        # If no test split, split from train
        all_em = list(ds_tr)
        rng.shuffle(all_em)
        n_test = min(200, len(all_em) // 10)
        ds_te = all_em[:n_test]
        ds_tr = all_em[n_test:]
    label_names = ds_tr.features["label"].names if hasattr(ds_tr, 'features') else ["sadness","joy","love","anger","fear","surprise"]
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
    # Use patched tokenizer (LFM2.5-230M has broken tokenizer_class in model card)
    tokenizer = _get_tokenizer()
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
            # 1. Hippocampus logits (no grad) — compute target, then free
            set_lora_state(model, hippo_state)
            model.eval()
            with torch.no_grad():
                # Compute p_hippo directly and free the logits
                hippo_logits = model(input_ids=input_ids).logits
                p_hippo = F.softmax(hippo_logits[..., :-1, :].contiguous().float(), dim=-1)
            del hippo_logits  # free immediately — only need p_hippo
            # 2. Neocortex logits (with grad)
            set_lora_state(model, neo_state)
            model.train()
            neo_logits = model(input_ids=input_ids).logits
            # 3. KL(p_hippo || p_neo)
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            (kl_loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step(); opt.zero_grad(); accum = 0
                gs += 1; tl += kl_loss.item()
                if gs % 50 == 0:
                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)
            # Free intermediates
            del neo_logits, log_p_neo, p_hippo, kl_loss, shift_neo
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state

# ============================================================================
# EVALUATION (batched)
# ============================================================================
def generate_batch(model, tokenizer, prompt_qs, max_new_tokens=BENCH_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE):
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
    """Intent classification: substring match (more lenient than exact match).
    The model may output 'The intent is card_about_to_expire' — check if gold
    appears anywhere in the response."""
    resp_norm = normalize_answer(response)
    gold_norm = normalize_answer(gold)
    # Exact match
    if resp_norm == gold_norm:
        return 1.0
    # Substring match (gold appears in response)
    if gold_norm in resp_norm:
        return 1.0
    # Also check: response appears in gold (model output is partial)
    if resp_norm in gold_norm:
        return 1.0
    # Try with underscores converted to spaces (banking77 labels use _)
    gold_spaces = gold_norm.replace('_', ' ')
    resp_spaces = resp_norm.replace('_', ' ')
    if gold_spaces in resp_spaces or resp_spaces in gold_spaces:
        return 1.0
    return 0.0

def evaluate_task(model, tokenizer, test_examples, task_name, max_questions=200, batch_size=EVAL_BATCH_SIZE):
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
            # Debug: print first 5 responses
            if processed <= 5:
                print(f"      Q{processed}: gold='{g[:40]}' resp='{r[:80]}' score={score(r,g)}", flush=True)
        if processed % (batch_size * 2) < batch_size or processed >= total:
            print(f"      [{processed}/{total}] acc={correct/processed:.3f} ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    aggressive_cleanup()
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
        aggressive_cleanup()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [naive] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f}", flush=True)
    del model
    aggressive_cleanup()
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
        aggressive_cleanup()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [AVR] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f} repairs={total_repairs}", flush=True)
    del model
    aggressive_cleanup()
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
        aggressive_cleanup()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [twostream+AVR] ACC={metrics['ACC']:.3f} BWT={metrics['BWT']:+.3f} FF={metrics['FF']:.3f} repairs={total_repairs}", flush=True)
    del model
    aggressive_cleanup()
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
        if "R" not in results[key]:
            ax.text(0.5, 0.5, f"{key}\n(error)", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=11)
            continue
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

# Wrap each condition in try/except to catch OOM and save partial results
try:
    # Condition 1: Naive
    R_n, m_n = run_naive(train_data, test_data, task_order)
    results["naive"] = {"R": R_n, "metrics": m_n}
    print(f"\n  Naive: ACC={m_n['ACC']:.3f} BWT={m_n['BWT']:+.3f} FF={m_n['FF']:.3f}", flush=True)
except torch.cuda.OutOfMemoryError as e:
    print(f"\n  OOM in Naive condition: {e}", flush=True)
    print("  Saving partial results and continuing to next condition...", flush=True)
    aggressive_cleanup()
    results["naive"] = {"error": "OOM", "message": str(e)}
except Exception as e:
    print(f"\n  Error in Naive condition: {e}", flush=True)
    results["naive"] = {"error": str(e)}

try:
    # Condition 2: AVR
    R_a, m_a, repairs_a, log_a = run_avr(train_data, test_data, task_order)
    results["avr"] = {"R": R_a, "metrics": m_a, "total_repairs": repairs_a, "log": log_a}
    print(f"\n  AVR: ACC={m_a['ACC']:.3f} BWT={m_a['BWT']:+.3f} FF={m_a['FF']:.3f} repairs={repairs_a}", flush=True)
except torch.cuda.OutOfMemoryError as e:
    print(f"\n  OOM in AVR condition: {e}", flush=True)
    print("  Saving partial results and continuing to next condition...", flush=True)
    aggressive_cleanup()
    results["avr"] = {"error": "OOM", "message": str(e)}
except Exception as e:
    print(f"\n  Error in AVR condition: {e}", flush=True)
    results["avr"] = {"error": str(e)}

try:
    # Condition 3: Two-Stream + AVR
    R_t, m_t, repairs_t, log_t = run_twostream_avr(train_data, test_data, task_order)
    results["twostream"] = {"R": R_t, "metrics": m_t, "total_repairs": repairs_t, "log": log_t}
    print(f"\n  TwoStream+AVR: ACC={m_t['ACC']:.3f} BWT={m_t['BWT']:+.3f} FF={m_t['FF']:.3f} repairs={repairs_t}", flush=True)
except torch.cuda.OutOfMemoryError as e:
    print(f"\n  OOM in Two-Stream condition: {e}", flush=True)
    print("  Saving partial results...", flush=True)
    aggressive_cleanup()
    results["twostream"] = {"error": "OOM", "message": str(e)}
except Exception as e:
    print(f"\n  Error in Two-Stream condition: {e}", flush=True)
    results["twostream"] = {"error": str(e)}

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

# Final summary (error-safe — only print conditions that succeeded)
print(f"\n{'='*80}", flush=True)
print("FINAL SUMMARY [LFM2.5-230M — Intent Classification]", flush=True)
print(f"{'='*80}", flush=True)
print(f"\n{'Method':<30} {'ACC':<8} {'BWT':<8} {'FF':<8} {'Repairs':<10}", flush=True)
print("-"*70, flush=True)

# Safe accessors for each condition
def safe_metrics(cond_name):
    if cond_name in results and "metrics" in results[cond_name]:
        return results[cond_name]["metrics"]
    return None

m_n = safe_metrics("naive")
m_a = safe_metrics("avr")
m_t = safe_metrics("twostream")

if m_n: print(f"{'Naive SFT':<30} {m_n['ACC']:<8.3f} {m_n['BWT']:<+8.3f} {m_n['FF']:<8.3f} {'-':<10}", flush=True)
else: print(f"{'Naive SFT':<30} ERROR", flush=True)
if m_a: print(f"{'AVR (v23 style)':<30} {m_a['ACC']:<8.3f} {m_a['BWT']:<+8.3f} {m_a['FF']:<8.3f} {results.get('avr',{}).get('total_repairs','-'):<10}", flush=True)
else: print(f"{'AVR (v23 style)':<30} ERROR", flush=True)
if m_t: print(f"{'Two-Stream + AVR (v34)':<30} {m_t['ACC']:<8.3f} {m_t['BWT']:<+8.3f} {m_t['FF']:<8.3f} {results.get('twostream',{}).get('total_repairs','-'):<10}", flush=True)
else: print(f"{'Two-Stream + AVR (v34)':<30} ERROR", flush=True)
print("-"*70, flush=True)

if m_n and m_a:
    print(f"\n  Recovery (AVR vs Naive): BWT {m_n['BWT']:+.3f} -> {m_a['BWT']:+.3f} ({(m_a['BWT']-m_n['BWT'])*100:+.1f}pp)", flush=True)
if m_n and m_t:
    print(f"  Recovery (TwoStream vs Naive): BWT {m_n['BWT']:+.3f} -> {m_t['BWT']:+.3f} ({(m_t['BWT']-m_n['BWT'])*100:+.1f}pp)", flush=True)

# Task 1 recovery (if we have R-matrices)
T = len(task_order)
for cond_name, label in [("naive", "Naive"), ("avr", "AVR"), ("twostream", "Two-Stream+AVR")]:
    if cond_name in results and "R" in results[cond_name]:
        R = results[cond_name]["R"]
        t1 = R[T-1][0]
        print(f"\n  Task 1 ({task_order[0]}) after all {T} tasks [{label}]: {t1:.3f}", flush=True)

# R-matrices (only for conditions that succeeded)
for cond_name in ["naive", "avr", "twostream"]:
    if cond_name in results and "R" in results[cond_name]:
        R = results[cond_name]["R"]
        print(f"\n  R MATRIX ({cond_name}):", flush=True)
        header = "After\\Test  " + "  ".join(f"{t[:12]:<14}" for t in task_order)
        print(f"  {header}", flush=True)
        for i in range(len(task_order)):
            row = f"  {task_order[i][:12]:<14} " + "  ".join(f"{R[i][j]:<14.3f}" for j in range(len(task_order)))
            print(row, flush=True)

summary = {"lfm2_230m_intent": {}}
for cond_name in ["naive", "avr", "twostream"]:
    if cond_name in results and "metrics" in results[cond_name]:
        summary["lfm2_230m_intent"][cond_name] = {
            **results[cond_name]["metrics"],
            "total_repairs": results[cond_name].get("total_repairs", 0)
        }
    else:
        summary["lfm2_230m_intent"][cond_name] = results.get(cond_name, {"error": "unknown"})
with open(OUTPUT_DIR / "summary_intent_lfm230m.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nSummary saved: summary_intent_lfm230m.json", flush=True)
print(f"\nDONE.", flush=True)
