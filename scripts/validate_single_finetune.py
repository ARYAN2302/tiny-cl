"""
Single-fine-tune validation: does AVR catch/repair drift on general capabilities
after ONE SFT task?

This tests the 95% pain: "I fine-tuned my model and it forgot how to do X."

Setup:
  1. Load Qwen3-1.7B
  2. Measure baseline on GSM8K (general math reasoning — proxy for "capabilities")
  3. SFT on MATH(algebra) — 5000 examples, 3 epochs
  4. Measure GSM8K again (did it drop?)
  5. Run AVR repair (PPL drift detection + weight interpolation toward pre-SFT snapshot)
  6. Measure GSM8K after repair (did it recover?)

Conditions:
  A. Naive: SFT on MATH → check GSM8K (no repair)
  B. AVR: SFT on MATH → detect drift → repair → check GSM8K

If GSM8K drops after SFT and AVR recovers it → the "your fine-tune broke your model"
pitch is validated for single-stage scenarios.

Runtime: ~1 hour on Kaggle T4 (230M model, small eval sets)
Model: LFM2.5-230M (small + fast + we have the tokenizer patch worked out)
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "datasets>=3.0.0",
    "accelerate>=1.0.0", "matplotlib", "sentencepiece",
    "protobuf", "packaging"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

# Patch transformers dynamo flag to avoid numpy crash on Kaggle
import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False

import json, time, random, math, gc, re, copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "LiquidAI/LFM2.5-230M"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "conv1d"]

TRAIN_LR = 2e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 3
BATCH_SIZE = 4
CONTEXT_LENGTH = 512
GRAD_ACCUM_STEPS = 4

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

BENCH_MAX_NEW_TOKENS = 200
EVAL_BATCH_SIZE = 8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# ============================================================================
# TOKENIZER (patched for LFM2 broken tokenizer_class)
# ============================================================================
_TOKENIZER = None
def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    import json
    from huggingface_hub import hf_hub_download
    cache_dir = OUTPUT_DIR / "tokenizer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "added_tokens.json"]:
        try: hf_hub_download(MODEL_ID, fname, local_dir=str(cache_dir))
        except: pass
    config_path = cache_dir / "tokenizer_config.json"
    if config_path.exists():
        with open(config_path) as f: config = json.load(f)
        if config.get("tokenizer_class") == "TokenizersBackend":
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            with open(config_path, "w") as f: json.dump(config, f, indent=2)
            print("  Patched tokenizer_class", flush=True)
    tok = AutoTokenizer.from_pretrained(str(cache_dir), use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        template_path = cache_dir / "chat_template.jinja"
        if template_path.exists():
            with open(template_path) as f: tok.chat_template = f.read()
            print(f"  Loaded chat_template ({len(tok.chat_template)} chars)", flush=True)
    _TOKENIZER = tok
    return tok

def format_prompt(question):
    tok = get_tokenizer()
    messages = [{"role": "user", "content": question}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_example(question, answer):
    tok = get_tokenizer()
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text + tok.eos_token

# ============================================================================
# DATA — GSM8K (probe) + MATH algebra (fine-tune task)
# ============================================================================
def load_gsm8k_probe(n=100):
    """Load GSM8K as the general-capability probe set."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        prompt = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{q}"
        pairs.append((prompt, a, gold))
    return pairs

def load_math_algebra_train(n=5000):
    """Load MATH algebra as the fine-tune task."""
    from datasets import load_dataset
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    except:
        ds = load_dataset("lighteval/MATH", "algebra", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["problem"]; sol = ex["solution"]
        prompt = f"Solve the math problem. End with the final answer in \\boxed{{}}.\n\n{q}"
        m = re.findall(r'\\boxed\{([^}]+)\}', sol)
        gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((prompt, sol, gold))
    return pairs

# ============================================================================
# MODEL + LoRA
# ============================================================================
def create_model():
    print(f"  Loading {MODEL_ID}...", flush=True)
    tokenizer = get_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                             target_modules=LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}", flush=True)
    return model, tokenizer

# ============================================================================
# LoRA STATE + AVR
# ============================================================================
def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

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

def repair_toward_snapshot(model, snapshot, alpha=REPAIR_ALPHA):
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name and name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(DEVICE))
            n += 1
    return n

# ============================================================================
# TRAINING
# ============================================================================
class TextDataset(Dataset):
    def __init__(self, token_ids, ctx_len):
        self.token_ids = token_ids
        self.ctx_len = ctx_len
        self.n_chunks = max(1, len(token_ids) // ctx_len)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.ctx_len; e = s + self.ctx_len
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}

def train_lora(model, tokenizer, examples, tag="sft"):
    tok = get_tokenizer()
    all_tokens = []
    for prompt_q, train_answer, gold in examples:
        text = format_example(prompt_q, train_answer)
        all_tokens.extend(tok.encode(text, add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=TRAIN_LR, weight_decay=TRAIN_WD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    gs, tl = 0, 0.0; t0 = time.time(); accum = 0; opt.zero_grad()
    for epoch in range(TASK_EPOCHS):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            (out.loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step(); opt.zero_grad(); accum = 0
                gs += 1; tl += out.loss.item()
                if gs % 50 == 0:
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)

# ============================================================================
# EVALUATION (batched)
# ============================================================================
def generate_batch(model, tokenizer, prompts, batch_size=EVAL_BATCH_SIZE):
    tok = get_tokenizer()
    results = []
    gc_was = getattr(model, "gradient_checkpointing", False)
    if gc_was:
        try: model.gradient_checkpointing_disable()
        except: pass
    model.eval()
    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            texts = [format_prompt(q) for q in batch]
            tok.padding_side = "left"
            inputs = tok(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=BENCH_MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tok.pad_token_id, temperature=1.0)
            for out in outputs:
                input_len = inputs["input_ids"].shape[1]
                results.append(tok.decode(out[input_len:], skip_special_tokens=True).strip())
    finally:
        if gc_was:
            try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
            except: pass
    return results

def normalize_math(s):
    s = s.strip().replace('$','').replace('\\','').replace('!','').replace(',','')
    s = s.replace('{','').replace('}','').replace(' ','')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except: return s.lower()

def extract_answer(response):
    response = response.strip()
    matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    if matches: return matches[-1].strip()
    m = re.search(r'####\s*(-?[\d,.]+)', response)
    if m: return m.group(1).replace(",", "").strip()
    m = re.search(r'(?:final answer|answer)\s*:?\s*\**\s*([^\n*]+)', response, re.IGNORECASE)
    if m:
        c = m.group(1).strip().rstrip('*').strip()
        nums = re.findall(r'-?\d[\d,.]*', c)
        if nums: return nums[-1].replace(",", "").strip()
    numbers = re.findall(r'-?\d[\d,.]*', response)
    if numbers: return numbers[-1].replace(",", "").strip()
    return response[:50] if response else ""

def evaluate_gsm8k(model, tokenizer, probe_set, label=""):
    print(f"    Eval GSM8K {label} ({len(probe_set)} Qs)...", flush=True)
    prompts = [ex[0] for ex in probe_set]
    golds = [ex[2] for ex in probe_set]
    correct = 0; t0 = time.time()
    for i in range(0, len(prompts), EVAL_BATCH_SIZE):
        batch_p = prompts[i:i+EVAL_BATCH_SIZE]
        batch_g = golds[i:i+EVAL_BATCH_SIZE]
        responses = generate_batch(model, tokenizer, batch_p)
        for r, g in zip(responses, batch_g):
            if normalize_math(extract_answer(r)) == normalize_math(g):
                correct += 1
    acc = correct / len(probe_set)
    print(f"    GSM8K {label}: {correct}/{len(probe_set)} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return acc

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("SINGLE-FINE-TUNE VALIDATION", flush=True)
print(f"Model: {MODEL_ID} | Seed: {SEED}", flush=True)
print(f"Probe: GSM8K (general math reasoning)", flush=True)
print(f"Fine-tune task: MATH(algebra)", flush=True)
print(f"AVR: threshold={DRIFT_THRESHOLD}, alpha={REPAIR_ALPHA}, max_steps={MAX_REPAIR_STEPS}", flush=True)
print("="*70, flush=True)

# Load data
print("\nLoading data...", flush=True)
gsm8k_probe = load_gsm8k_probe(n=100)
math_train = load_math_algebra_train(n=5000)
print(f"  GSM8K probe: {len(gsm8k_probe)} questions", flush=True)
print(f"  MATH algebra train: {len(math_train)} examples", flush=True)

results = {}

# === BASELINE: measure GSM8K on base model (no fine-tune) ===
print(f"\n{'='*60}\n  BASELINE: GSM8K on base model\n{'='*60}", flush=True)
model, tokenizer = create_model()
baseline_acc = evaluate_gsm8k(model, tokenizer, gsm8k_probe, "baseline")
baseline_ppl = compute_ppl(model, tokenizer, gsm8k_probe)
results["baseline"] = {"gsm8k_acc": baseline_acc, "gsm8k_ppl": baseline_ppl}
print(f"  Baseline: GSM8K acc={baseline_acc:.3f}, PPL={baseline_ppl:.2f}", flush=True)

# Take snapshot of base model LoRA state (for repair target)
base_snapshot = get_lora_state(model)
print(f"  Snapshot taken (base model state)", flush=True)

# === CONDITION A: NAIVE — SFT on MATH, check GSM8K (no repair) ===
print(f"\n{'='*60}\n  CONDITION A: Naive SFT on MATH(algebra) — no repair\n{'='*60}", flush=True)
train_lora(model, tokenizer, math_train, tag="naive-sft")
naive_acc = evaluate_gsm8k(model, tokenizer, gsm8k_probe, "after-naive-sft")
naive_ppl = compute_ppl(model, tokenizer, gsm8k_probe)
results["naive"] = {"gsm8k_acc": naive_acc, "gsm8k_ppl": naive_ppl}
print(f"  Naive: GSM8K acc={naive_acc:.3f} (was {baseline_acc:.3f}, delta={naive_acc-baseline_acc:+.3f})", flush=True)
print(f"  Naive: GSM8K PPL={naive_ppl:.2f} (was {baseline_ppl:.2f}, ratio={naive_ppl/baseline_ppl:.2f}x)", flush=True)

del model; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# === CONDITION B: AVR — SFT on MATH, detect drift, repair, check GSM8K ===
print(f"\n{'='*60}\n  CONDITION B: AVR — SFT on MATH(algebra) + drift detect + repair\n{'='*60}", flush=True)
model, tokenizer = create_model()

# Verify baseline matches
avr_baseline_acc = evaluate_gsm8k(model, tokenizer, gsm8k_probe, "avr-baseline")
avr_base_snapshot = get_lora_state(model)

# SFT on MATH
train_lora(model, tokenizer, math_train, tag="avr-sft")
post_sft_ppl = compute_ppl(model, tokenizer, gsm8k_probe)
post_sft_acc = evaluate_gsm8k(model, tokenizer, gsm8k_probe, "after-sft-pre-repair")

print(f"\n  Post-SFT: GSM8K acc={post_sft_acc:.3f} (was {avr_baseline_acc:.3f})", flush=True)
print(f"  Post-SFT: GSM8K PPL={post_sft_ppl:.2f} (was {baseline_ppl:.2f}, ratio={post_sft_ppl/baseline_ppl:.2f}x)", flush=True)

# VERIFY: check drift
drift_ratio = post_sft_ppl / baseline_ppl if baseline_ppl > 0 else 1.0
drifted = drift_ratio > DRIFT_THRESHOLD
print(f"\n  [VERIFY] PPL ratio = {drift_ratio:.2f}x (threshold {DRIFT_THRESHOLD}x)", flush=True)
if drifted:
    print(f"  [VERIFY] DRIFT DETECTED — GSM8K PPL rose {drift_ratio:.2f}x above baseline", flush=True)
else:
    print(f"  [VERIFY] No drift detected (ratio below threshold)", flush=True)

# REPAIR: if drifted, repair toward base snapshot
repair_steps = 0
if drifted:
    print(f"\n  [REPAIR] Repairing toward base model snapshot...", flush=True)
    for step in range(MAX_REPAIR_STEPS):
        n_adj = repair_toward_snapshot(model, avr_base_snapshot)
        repair_steps += 1
        repair_ppl = compute_ppl(model, tokenizer, gsm8k_probe)
        repair_ratio = repair_ppl / baseline_ppl if baseline_ppl > 0 else 1.0
        print(f"    [REPAIR] Step {step+1}: {n_adj} params, PPL={repair_ppl:.2f} (ratio {repair_ratio:.2f}x)", flush=True)
        if repair_ratio <= DRIFT_THRESHOLD:
            print(f"  [REPAIR] Converged at step {step+1}", flush=True)
            break
    else:
        print(f"  [REPAIR] Max steps ({MAX_REPAIR_STEPS}) reached", flush=True)

# Final eval after repair
avr_final_acc = evaluate_gsm8k(model, tokenizer, gsm8k_probe, "after-repair")
avr_final_ppl = compute_ppl(model, tokenizer, gsm8k_probe)
results["avr"] = {
    "gsm8k_acc_baseline": avr_baseline_acc,
    "gsm8k_acc_post_sft": post_sft_acc,
    "gsm8k_acc_after_repair": avr_final_acc,
    "gsm8k_ppl_baseline": baseline_ppl,
    "gsm8k_ppl_post_sft": post_sft_ppl,
    "gsm8k_ppl_after_repair": avr_final_ppl,
    "drift_detected": drifted,
    "drift_ratio": drift_ratio,
    "repair_steps": repair_steps,
}

# === SUMMARY ===
print(f"\n{'='*70}", flush=True)
print("SINGLE-FINE-TUNE VALIDATION RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n  Model: {MODEL_ID}", flush=True)
print(f"  Probe: GSM8K (100 questions)", flush=True)
print(f"  Fine-tune: MATH(algebra), 5000 examples, 3 epochs", flush=True)
print(f"\n  {'Metric':<30} {'Baseline':<12} {'Naive SFT':<12} {'AVR':<12}", flush=True)
print(f"  {'-'*66}", flush=True)
print(f"  {'GSM8K accuracy':<30} {baseline_acc:<12.3f} {naive_acc:<12.3f} {avr_final_acc:<12.3f}", flush=True)
print(f"  {'GSM8K PPL':<30} {baseline_ppl:<12.2f} {naive_ppl:<12.2f} {avr_final_ppl:<12.2f}", flush=True)
print(f"  {'Drift detected':<30} {'—':<12} {'—':<12} {str(drifted):<12}", flush=True)
print(f"  {'Repair steps':<30} {'—':<12} {'—':<12} {repair_steps:<12}", flush=True)
print(f"  {'-'*66}", flush=True)
print(f"\n  Delta (Naive vs Baseline):     acc {naive_acc-baseline_acc:+.3f}, PPL {naive_ppl-baseline_ppl:+.2f}", flush=True)
print(f"  Delta (AVR vs Baseline):       acc {avr_final_acc-baseline_acc:+.3f}, PPL {avr_final_ppl-baseline_ppl:+.2f}", flush=True)
print(f"  Recovery (AVR vs Naive):       acc {avr_final_acc-naive_acc:+.3f}", flush=True)
print(f"\n  Verdict:", flush=True)
if drifted and avr_final_acc > naive_acc:
    print(f"  ✅ AVR detected drift ({drift_ratio:.2f}x) and recovered GSM8K accuracy", flush=True)
    print(f"     Naive: {baseline_acc:.3f} → {naive_acc:.3f} (delta {naive_acc-baseline_acc:+.3f})", flush=True)
    print(f"     AVR:   {avr_baseline_acc:.3f} → {avr_final_acc:.3f} (delta {avr_final_acc-avr_baseline_acc:+.3f})", flush=True)
    print(f"     The 'your fine-tune broke your model' pitch is VALIDATED.", flush=True)
elif not drifted:
    print(f"  ⚠ No drift detected (PPL ratio {drift_ratio:.2f}x below threshold {DRIFT_THRESHOLD}x)", flush=True)
    print(f"     Single-fine-tune may not produce enough drift for AVR to catch.", flush=True)
    print(f"     The pitch stays as 'task streams' (multi-task validated).", flush=True)
else:
    print(f"  ❌ Drift detected but repair didn't recover accuracy.", flush=True)
    print(f"     The mechanism needs tuning for single-fine-tune scenarios.", flush=True)

with open(OUTPUT_DIR / "single_finetune_validation.json", "w") as f:
    json.dump({"model": MODEL_ID, "seed": SEED, "results": results}, f, indent=2, default=str)
print(f"\nResults saved: {OUTPUT_DIR}/single_finetune_validation.json", flush=True)
print(f"\nDONE.", flush=True)
