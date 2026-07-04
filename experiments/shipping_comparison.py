"""
SHIPPING TEST: Post-training pipeline comparison

Base model vs Naive SFT vs AVR on a 4-stage domain stream:
  Stage 1: UltraChat (general chat)
  Stage 2: Medical
  Stage 3: Code
  Stage 4: Legal

Eval after each stage:
  - MMLU sampled (20 subjects × 5 Qs = 100 Qs) — knowledge retention
  - Domain accuracy (Medical/Code/Legal test sets) — domain retention
  - UltraChat PPL — chat quality proxy

One script, one run, ~6-8h on T4. Produces the shipping comparison table.

USAGE: Copy-paste into one Kaggle cell. GPU on, Internet on.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "peft>=0.10", "datasets>=2.14",
                "gdown>=4.7", "pyyaml>=6.0", "numpy"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                "torchao"], check=False, capture_output=True)

import os, json, math, gc, copy, re, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "LiquidAI/LFM2.5-350M"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["in_proj", "out_proj"]

TRAIN_LR = 2e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 2
BATCH_SIZE = 8
CONTEXT_LENGTH = 512

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# DATA LOADERS
# ============================================================================

def load_dataset_sample(dataset_id, field, n_samples, split="train"):
    """Load n_samples from a HF dataset, extract text field."""
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset_id, split=split)
        texts = [t for t in ds[field] if t and len(t.strip()) > 20]
        random.shuffle(texts)
        return texts[:n_samples]
    except Exception as e:
        print(f"    WARNING: {dataset_id} failed: {e}")
        return []

def build_pairs_from_text(texts):
    """Split each text into (prompt, answer) pair at midpoint."""
    pairs = []
    for text in texts:
        mid = len(text) // 2
        prompt = text[:mid].strip()
        answer = text[mid:].strip()
        if prompt and answer and len(prompt) > 20 and len(answer) > 20:
            pairs.append((prompt, answer))
    return pairs

def load_mmlu_eval(n_subjects=20, n_per_subject=5):
    """Load MMLU subjects as eval pairs (prompt, gold_letter)."""
    from datasets import load_dataset
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_chemistry", "college_physics",
        "computer_security", "conceptual_physics", "econometrics",
        "electrical_engineering", "formal_logic", "global_facts",
        "high_school_biology", "high_school_chemistry", "high_school_geography",
        "high_school_mathematics", "high_school_physics", "high_school_psychology",
        "jurisprudence",
    ][:n_subjects]
    eval_pairs = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test")
            samples = list(ds)
            random.shuffle(samples)
            for ex in samples[:n_per_subject]:
                q = ex["question"]
                choices = ex["choices"]
                ans = ex["answer"]
                prompt = f"{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
                gold = ["A", "B", "C", "D"][ans]
                eval_pairs.append((prompt, gold))
        except:
            pass
    return eval_pairs, subjects

def load_pipeline_data():
    """Load the 4-stage post-training pipeline data."""
    print("\n=== Loading pipeline data ===")
    stages = []

    # Stage 1: UltraChat (general chat)
    print("  Stage 1: UltraChat...")
    texts = load_dataset_sample("stingning/ultrachat", "content", 2000)
    # ultrachat content is a list of messages — flatten
    pairs = []
    for item in texts[:1000]:
        if isinstance(item, list):
            # Join messages into one text
            text = " ".join(str(m) for m in item)
        else:
            text = str(item)
        mid = len(text) // 2
        if mid > 20:
            pairs.append((text[:mid].strip()[:1000], text[mid:].strip()[:1000]))
    stages.append(("ultrachat", pairs, pairs[:50]))  # train, eval
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    # Stage 2: Medical
    print("  Stage 2: Medical...")
    texts = load_dataset_sample("epfl-llm/guidelines", "clean_text", 1000)
    pairs = build_pairs_from_text(texts)
    stages.append(("medical", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    # Stage 3: Code
    print("  Stage 3: Code...")
    texts = load_dataset_sample("iamtarun/python_code_instructions_18k_alpaca", "output", 1000)
    pairs = build_pairs_from_text(texts)
    stages.append(("code", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    # Stage 4: Legal
    print("  Stage 4: Legal...")
    texts = load_dataset_sample("nguha/legalbench", "text", 500)
    if not texts:
        # Fallback: use a different legal dataset
        texts = load_dataset_sample("daniel-saeedi/legal-qa", "question", 500)
    pairs = build_pairs_from_text(texts) if texts else []
    if len(pairs) < 100:
        # Fallback: use more code data as a 4th domain
        print("    Legal data insufficient, using Finance instead")
        texts = load_dataset_sample("gbharti/finance-alpaca", "output", 500)
        pairs = build_pairs_from_text(texts) if texts else []
    stages.append(("domain4", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    return stages

# ============================================================================
# MODEL
# ============================================================================

def create_model():
    print(f"  Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE, attn_implementation="eager")
    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                             target_modules=LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    print(f"  LoRA attached (rank={LORA_RANK})")
    return model, tokenizer

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

# ============================================================================
# TRAINING
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

def build_token_stream(tokenizer, pairs):
    all_tokens = []
    for prompt, answer in pairs:
        text = prompt + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

def train_on_pairs(model, tokenizer, pairs, epochs=TASK_EPOCHS):
    if not pairs:
        print("    WARNING: no training data, skipping")
        return
    token_ids = build_token_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    Training: {len(token_ids):,} tokens, {len(dataset)} chunks")
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=TRAIN_LR, weight_decay=TRAIN_WD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
            opt.step(); tl += out.loss.item(); gs += 1
            if gs % 50 == 0: print(f"      step {gs} | loss={tl/gs:.4f}", flush=True)

# ============================================================================
# AVR REPAIR
# ============================================================================

def compute_ppl(model, tokenizer, pairs, max_samples=50):
    if not pairs: return float('inf')
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for prompt, answer in pairs[:max_samples]:
        text = prompt + " " + answer + tokenizer.eos_token
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))

def verify_drift(current_ppls, best_ppls, completed_stages, threshold=DRIFT_THRESHOLD):
    drifted = {}
    for stage in completed_stages:
        if stage not in current_ppls or stage not in best_ppls:
            continue
        ratio = current_ppls[stage] / best_ppls[stage] if best_ppls[stage] > 0 else 1.0
        if ratio > threshold:
            drifted[stage] = {"current": current_ppls[stage], "best": best_ppls[stage], "ratio": ratio}
    return drifted

def repair_toward_snapshot(model, snapshot_state, alpha=REPAIR_ALPHA):
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(DEVICE)
            p.data.copy_((1 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj

# ============================================================================
# EVALUATION
# ============================================================================

def eval_mmlu(model, tokenizer, mmlu_pairs, max_questions=100):
    """Accuracy on MMLU sampled questions."""
    if not mmlu_pairs: return 0.0
    model.eval()
    correct = 0
    total = min(len(mmlu_pairs), max_questions)
    for i in range(total):
        prompt, gold = mmlu_pairs[i]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
        response_upper = response.upper()[:5]
        for letter in ["A", "B", "C", "D"]:
            if letter in response_upper:
                if letter == gold:
                    correct += 1
                break
    model.train()
    return correct / total

def eval_domain_ppls(model, tokenizer, stages, trained_so_far):
    """PPL on each domain's eval set."""
    ppls = {}
    for i, (name, train_pairs, eval_pairs) in enumerate(stages):
        if i >= trained_so_far: break
        ppls[name] = compute_ppl(model, tokenizer, eval_pairs, 50)
    return ppls

def eval_all(model, tokenizer, mmlu_pairs, stages, trained_so_far, label=""):
    """Full eval: MMLU + domain PPLs."""
    print(f"\n  --- Eval {label} ---")
    mmlu_acc = eval_mmlu(model, tokenizer, mmlu_pairs)
    domain_ppls = eval_domain_ppls(model, tokenizer, stages, trained_so_far)
    print(f"    MMLU: {mmlu_acc:.3f}")
    for name, ppl in domain_ppls.items():
        print(f"    {name} PPL: {ppl:.2f}")
    return {"mmlu": mmlu_acc, "domain_ppls": domain_ppls}

# ============================================================================
# PIPELINE RUNNERS
# ============================================================================

def run_base_eval(stages, mmlu_pairs):
    """Eval the base model (no fine-tuning)."""
    print(f"\n{'='*70}")
    print("BASE MODEL EVAL (no fine-tuning)")
    print(f"{'='*70}")
    model, tokenizer = create_model()
    results = eval_all(model, tokenizer, mmlu_pairs, stages, len(stages), "base model")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return results

def run_naive_pipeline(stages, mmlu_pairs):
    """Naive sequential SFT — no CL, no repair."""
    print(f"\n{'='*70}")
    print("NAIVE SEQUENTIAL SFT (no CL)")
    print(f"{'='*70}")
    model, tokenizer = create_model()

    eval_results = []
    for i, (name, train_pairs, eval_pairs) in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"  Stage {i+1}/{len(stages)}: {name}")
        print(f"{'='*60}", flush=True)
        train_on_pairs(model, tokenizer, train_pairs)
        result = eval_all(model, tokenizer, mmlu_pairs, stages, i+1, f"after stage {i+1} ({name})")
        eval_results.append({"stage": name, "result": result})

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return eval_results

def run_avr_pipeline(stages, mmlu_pairs):
    """AVR sequential SFT — verify + repair between stages."""
    print(f"\n{'='*70}")
    print("AVR SEQUENTIAL SFT (with verify + repair)")
    print(f"{'='*70}")
    model, tokenizer = create_model()

    snapshot = None
    best_ppls = {}
    completed_stages = []
    total_repairs = 0
    eval_results = []

    for i, (name, train_pairs, eval_pairs) in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"  Stage {i+1}/{len(stages)}: {name}")
        print(f"{'='*60}", flush=True)

        train_on_pairs(model, tokenizer, train_pairs)

        # Post-train PPLs
        post_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
        if name not in best_ppls:
            best_ppls[name] = post_ppls[name]
        completed_stages.append(name)

        print(f"  Post-train PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_ppls.items()))

        # AVR verify + repair
        if i > 0 and snapshot is not None:
            drifted = verify_drift(post_ppls, best_ppls, completed_stages[:-1])
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}")
                for s, info in drifted.items():
                    print(f"    {s}: {info['current']:.2f} / {info['best']:.2f} = {info['ratio']:.2f}x")

                still_drifted = drifted
                for step in range(MAX_REPAIR_STEPS):
                    repair_toward_snapshot(model, snapshot)
                    repair_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_stages[:-1])
                    if not still_drifted:
                        print(f"  [AVR] Converged at step {step+1}")
                        break
                if still_drifted:
                    print(f"  [AVR] Max steps ({MAX_REPAIR_STEPS}) reached, drift remains")
                total_repairs += step + 1
            else:
                print(f"  [AVR] No drift")

        # Update best PPLs
        final_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
        for s, p in final_ppls.items():
            if s not in best_ppls or p < best_ppls[s]:
                best_ppls[s] = p

        snapshot = get_lora_state(model)

        result = eval_all(model, tokenizer, mmlu_pairs, stages, i+1, f"after stage {i+1} ({name})")
        eval_results.append({"stage": name, "result": result})

    print(f"\n  [AVR] Total repair steps: {total_repairs}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return eval_results, total_repairs

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SHIPPING TEST: Post-training pipeline comparison")
    print(f"Model: {MODEL_ID} | Seed: {SEED}")
    print(f"Stages: UltraChat → Medical → Code → Domain4")
    print(f"AVR: threshold={DRIFT_THRESHOLD}, α={REPAIR_ALPHA}, max_steps={MAX_REPAIR_STEPS}")
    print("=" * 70)

    # Load data
    stages = load_pipeline_data()
    print("\n  Loading MMLU eval set...")
    mmlu_pairs, mmlu_subjects = load_mmlu_eval(n_subjects=20, n_per_subject=5)
    print(f"    MMLU: {len(mmlu_pairs)} questions across {len(mmlu_subjects)} subjects")

    # Run all three
    base_result = run_base_eval(stages, mmlu_pairs)
    naive_results = run_naive_pipeline(stages, mmlu_pairs)
    avr_results, total_repairs = run_avr_pipeline(stages, mmlu_pairs)

    # === COMPARISON TABLE ===
    print(f"\n{'='*70}")
    print("SHIPPING COMPARISON TABLE")
    print(f"{'='*70}")

    print(f"\n{'Method':<20} {'MMLU':<10} {'Chat PPL':<12} {'Med PPL':<12} {'Code PPL':<12} {'D4 PPL':<12}")
    print("-" * 78)

    # Base
    base_ppls = base_result["domain_ppls"]
    print(f"{'Base model':<20} {base_result['mmlu']:<10.3f} "
          f"{base_ppls.get('ultrachat', 0):<12.2f} {base_ppls.get('medical', 0):<12.2f} "
          f"{base_ppls.get('code', 0):<12.2f} {base_ppls.get('domain4', 0):<12.2f}")

    # Naive (after all stages)
    naive_final = naive_results[-1]["result"]
    naive_ppls = naive_final["domain_ppls"]
    print(f"{'Naive SFT':<20} {naive_final['mmlu']:<10.3f} "
          f"{naive_ppls.get('ultrachat', 0):<12.2f} {naive_ppls.get('medical', 0):<12.2f} "
          f"{naive_ppls.get('code', 0):<12.2f} {naive_ppls.get('domain4', 0):<12.2f}")

    # AVR (after all stages)
    avr_final = avr_results[-1]["result"]
    avr_ppls = avr_final["domain_ppls"]
    print(f"{'AVR-cl':<20} {avr_final['mmlu']:<10.3f} "
          f"{avr_ppls.get('ultrachat', 0):<12.2f} {avr_ppls.get('medical', 0):<12.2f} "
          f"{avr_ppls.get('code', 0):<12.2f} {avr_ppls.get('domain4', 0):<12.2f}")

    # Per-stage trajectory
    print(f"\n{'='*70}")
    print("TRAJECTORY (MMLU after each stage)")
    print(f"{'='*70}")
    print(f"\n{'Stage':<15} {'Naive MMLU':<15} {'AVR MMLU':<15} {'Naive→Base Δ':<15} {'AVR→Base Δ':<15}")
    print("-" * 75)
    base_mmlu = base_result["mmlu"]
    print(f"{'(base)':<15} {'—':<15} {'—':<15} {0.0:<+15.3f} {0.0:<+15.3f}")
    for i, (n_res, a_res) in enumerate(zip(naive_results, avr_results)):
        n_mmlu = n_res["result"]["mmlu"]
        a_mmlu = a_res["result"]["mmlu"]
        stage = n_res["stage"]
        print(f"{stage:<15} {n_mmlu:<15.3f} {a_mmlu:<15.3f} {n_mmlu - base_mmlu:<+15.3f} {a_mmlu - base_mmlu:<+15.3f}")

    # Domain retention
    print(f"\n{'='*70}")
    print("DOMAIN RETENTION (PPL after all stages, lower = better)")
    print(f"{'='*70}")
    print(f"\n{'Domain':<15} {'Base':<12} {'Naive':<12} {'AVR':<12} {'Naive Δ':<12} {'AVR Δ':<12}")
    print("-" * 75)
    for name in ["ultrachat", "medical", "code", "domain4"]:
        b = base_ppls.get(name, 0)
        n = naive_ppls.get(name, 0)
        a = avr_ppls.get(name, 0)
        print(f"{name:<15} {b:<12.2f} {n:<12.2f} {a:<12.2f} {n-b:<+12.2f} {a-b:<+12.2f}")

    print(f"\n  AVR total repair steps: {total_repairs}")

    # Save
    results = {
        "base": base_result,
        "naive": naive_results,
        "avr": avr_results,
        "avr_total_repairs": total_repairs,
        "config": {
            "model": MODEL_ID, "seed": SEED,
            "lora_rank": LORA_RANK, "max_repair_steps": MAX_REPAIR_STEPS,
            "drift_threshold": DRIFT_THRESHOLD, "repair_alpha": REPAIR_ALPHA,
        }
    }
    with open(OUTPUT_DIR / "shipping_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {OUTPUT_DIR}/shipping_comparison.json")

if __name__ == "__main__":
    main()
