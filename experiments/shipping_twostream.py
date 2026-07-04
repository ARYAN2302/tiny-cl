"""
SHIPPING TEST 2: Two-Stream + AVR on the post-training pipeline

Same pipeline as shipping_comparison.py (UltraChat → Medical → Code → Finance).
Same eval (MMLU + domain PPLs after each stage).
Only runs condition C: Two-Stream + AVR.

Slot the results into the existing table:
  Base:      0.370 MMLU
  Naive:     0.230 MMLU
  AVR-alone: 0.290 MMLU
  TwoStream: (this run)

USAGE: Copy-paste into one Kaggle cell. GPU on, Internet on. ~5-6h on T4.
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora.layer import LoraLayer

MODEL_ID = "LiquidAI/LFM2.5-350M"
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SEED = 42
DATA_SEED = 42

LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["in_proj", "out_proj"]

TRAIN_LR = 2e-4
CONSOLIDATION_LR = 1e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 2
CONSOLIDATION_EPOCHS = 1
BATCH_SIZE = 8
CONTEXT_LENGTH = 512

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

random.seed(MODEL_SEED)
np.random.seed(MODEL_SEED)
torch.manual_seed(MODEL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MODEL_SEED)

data_rng = random.Random(DATA_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_TOKENS_PER_STAGE = 200000

# ============================================================================
# DATA (same as shipping_comparison.py)
# ============================================================================

def load_dataset_sample(dataset_id, field, n_samples, split="train"):
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset_id, split=split)
        texts = [t for t in ds[field] if t and (isinstance(t, str) and len(t.strip()) > 20)]
        data_rng.shuffle(texts)
        return texts[:n_samples]
    except Exception as e:
        print(f"    WARNING: {dataset_id} field '{field}' failed: {e}")
        return []

def load_ultrachat(n_samples=800):
    from datasets import load_dataset
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        texts = []
        for convo in ds:
            messages = convo.get("messages", [])
            if not messages: continue
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            text = "\n".join(parts)
            if len(text.strip()) > 50:
                texts.append(text)
        data_rng.shuffle(texts)
        return texts[:n_samples]
    except Exception as e:
        print(f"    WARNING: ultrachat_200k failed: {e}")
        return []

def build_pairs_from_text(texts):
    pairs = []
    for text in texts:
        if not text or len(text.strip()) < 40: continue
        text = text.strip()
        if len(text) < 1000:
            mid = len(text) // 2
            prompt = text[:mid].strip()
            answer = text[mid:].strip()
            if prompt and answer and len(prompt) > 20 and len(answer) > 20:
                pairs.append((prompt, answer))
        else:
            chunk_size = 800
            for i in range(0, len(text) - 100, chunk_size // 2):
                chunk = text[i:i + chunk_size]
                if len(chunk) < 100: break
                mid = len(chunk) // 2
                prompt = chunk[:mid].strip()
                answer = chunk[mid:].strip()
                if prompt and answer and len(prompt) > 20 and len(answer) > 20:
                    pairs.append((prompt, answer))
    return pairs

def cap_pairs_by_tokens(pairs, tokenizer, max_tokens=MAX_TOKENS_PER_STAGE):
    total_tokens = 0
    capped = []
    for prompt, answer in pairs:
        text = prompt + " " + answer
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if total_tokens + n_tokens > max_tokens: break
        capped.append((prompt, answer))
        total_tokens += n_tokens
    return capped

def load_mmlu_eval(n_subjects=20, n_per_subject=5):
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
            data_rng.shuffle(samples)
            for ex in samples[:n_per_subject]:
                q = ex["question"]
                choices = ex["choices"]
                ans = ex["answer"]
                prompt = f"{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
                gold = ["A", "B", "C", "D"][ans]
                eval_pairs.append((prompt, gold))
        except: pass
    return eval_pairs, subjects

def load_pipeline_data(tokenizer):
    print("\n=== Loading pipeline data ===")
    stages = []
    print("  Stage 1: UltraChat...")
    texts = load_ultrachat(800)
    pairs = []
    for text in texts:
        mid = len(text) // 2
        if mid > 50:
            pairs.append((text[:mid].strip()[:1500], text[mid:].strip()[:1500]))
    pairs = cap_pairs_by_tokens(pairs, tokenizer, MAX_TOKENS_PER_STAGE)
    stages.append(("ultrachat", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    print("  Stage 2: Medical...")
    texts = load_dataset_sample("epfl-llm/guidelines", "clean_text", 800)
    pairs = build_pairs_from_text(texts)
    pairs = cap_pairs_by_tokens(pairs, tokenizer, MAX_TOKENS_PER_STAGE)
    stages.append(("medical", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    print("  Stage 3: Code...")
    texts = load_dataset_sample("iamtarun/python_code_instructions_18k_alpaca", "output", 800)
    pairs = build_pairs_from_text(texts)
    pairs = cap_pairs_by_tokens(pairs, tokenizer, MAX_TOKENS_PER_STAGE)
    stages.append(("code", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")

    print("  Stage 4: Finance...")
    texts = load_dataset_sample("gbharti/finance-alpaca", "output", 800)
    if not texts or len(texts) < 100:
        print("    Finance data insufficient, using additional Code instead")
        texts = load_dataset_sample("iamtarun/python_code_instructions_18k_alpaca", "output", 800)
    pairs = build_pairs_from_text(texts)
    pairs = cap_pairs_by_tokens(pairs, tokenizer, MAX_TOKENS_PER_STAGE)
    stages.append(("finance", pairs, pairs[:50]))
    print(f"    {len(pairs)} train pairs, 50 eval pairs")
    return stages

# ============================================================================
# MODEL + LoRA STATE
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

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

def reset_lora_to_peft_init(model):
    """Reset LoRA to fresh PEFT init: A random, B zero."""
    for n, p in model.named_parameters():
        if "lora_A" in n:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()

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

def train_hippocampus(model, tokenizer, pairs, epochs=TASK_EPOCHS):
    """Train hippocampus LoRA in isolation (fresh init each task)."""
    token_ids = build_token_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    [hippo] Training: {len(token_ids):,} tokens, {len(dataset)} chunks")
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
            if gs % 50 == 0: print(f"      [hippo] step {gs} | loss={tl/gs:.4f}", flush=True)
    return gs, tl

def consolidate_to_neocortex(model, tokenizer, hippo_state, neo_state, pairs, epochs=CONSOLIDATION_EPOCHS):
    """Distill hippocampus → neocortex via KL divergence."""
    print(f"    [consolid] Distilling hippocampus → neocortex ({epochs} epoch)", flush=True)
    token_ids = build_token_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=CONSOLIDATION_LR, weight_decay=TRAIN_WD)

    gs, tl = 0, 0.0
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            # 1. Hippocampus logits (no grad)
            set_lora_state(model, hippo_state)
            model.eval()
            with torch.no_grad():
                hippo_out = model(input_ids=input_ids)
                hippo_logits = hippo_out.logits
            # 2. Neocortex logits (with grad)
            set_lora_state(model, neo_state)
            model.train()
            neo_out = model(input_ids=input_ids)
            neo_logits = neo_out.logits
            # 3. KL divergence
            shift_hippo = hippo_logits[..., :-1, :].contiguous()
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            p_hippo = F.softmax(shift_hippo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            opt.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
            opt.step()
            tl += kl_loss.item(); gs += 1
            if gs % 50 == 0: print(f"      [consolid] step {gs} | KL={tl/gs:.4f}", flush=True)
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state

# ============================================================================
# AVR REPAIR
# ============================================================================

def compute_ppl(model, tokenizer, pairs, max_samples=50):
    if not pairs or len(pairs) == 0: return float('nan')
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
    if total_tokens == 0: return float('nan')
    return math.exp(total_loss / total_tokens)

def verify_drift(current_ppls, best_ppls, completed_stages, threshold=DRIFT_THRESHOLD):
    drifted = {}
    for stage in completed_stages:
        if stage not in current_ppls or stage not in best_ppls: continue
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
                if letter == gold: correct += 1
                break
    model.train()
    return correct / total

def eval_domain_ppls(model, tokenizer, stages, trained_so_far):
    ppls = {}
    for i, (name, train_pairs, eval_pairs) in enumerate(stages):
        if i >= trained_so_far: break
        ppls[name] = compute_ppl(model, tokenizer, eval_pairs, 50)
    return ppls

def eval_all(model, tokenizer, mmlu_pairs, stages, trained_so_far, label=""):
    print(f"\n  --- Eval {label} ---")
    mmlu_acc = eval_mmlu(model, tokenizer, mmlu_pairs)
    domain_ppls = eval_domain_ppls(model, tokenizer, stages, trained_so_far)
    print(f"    MMLU: {mmlu_acc:.3f}")
    for name, ppl in domain_ppls.items():
        if math.isnan(ppl):
            print(f"    {name} PPL: N/A (no eval data)")
        else:
            print(f"    {name} PPL: {ppl:.2f}")
    return {"mmlu": mmlu_acc, "domain_ppls": domain_ppls}

# ============================================================================
# TWO-STREAM + AVR PIPELINE
# ============================================================================

def run_twostream_avr_pipeline(stages, mmlu_pairs):
    print(f"\n{'='*70}")
    print("TWO-STREAM + AVR PIPELINE")
    print(f"{'='*70}")
    model, tokenizer = create_model()

    neo_state = get_lora_state(model)  # starts at LoRA init
    snapshot = None
    best_ppls = {}
    completed_stages = []
    total_repairs = 0
    eval_results = []

    for i, (name, train_pairs, eval_pairs) in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"  Stage {i+1}/{len(stages)}: {name}")
        print(f"{'='*60}", flush=True)

        # 1. Snapshot neocortex
        neo_snapshot = copy.deepcopy(neo_state)
        print(f"  [twostream] Neocortex snapshot taken", flush=True)

        # 2. Reset hippocampus to fresh PEFT init
        reset_lora_to_peft_init(model)
        print(f"  [twostream] Hippocampus reset to fresh PEFT init", flush=True)

        # 3. Train hippocampus
        train_hippocampus(model, tokenizer, train_pairs)
        hippo_state = get_lora_state(model)
        print(f"  [twostream] Hippocampus trained", flush=True)

        # 4. Consolidate to neocortex
        set_lora_state(model, neo_state)
        neo_state = consolidate_to_neocortex(
            model, tokenizer, hippo_state, neo_state, train_pairs)
        print(f"  [twostream] Consolidation complete", flush=True)

        # 5. AVR check on neocortex
        set_lora_state(model, neo_state)
        post_consolid_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
        if name not in best_ppls:
            best_ppls[name] = post_consolid_ppls[name]
        completed_stages.append(name)

        print(f"  Post-consolid PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_consolid_ppls.items()))

        if i > 0:
            print(f"  Drift ratios vs best-seen:")
            for s in completed_stages[:-1]:
                if s in post_consolid_ppls and s in best_ppls:
                    ratio = post_consolid_ppls[s] / best_ppls[s] if best_ppls[s] > 0 else 1.0
                    drift_flag = " ← DRIFT" if ratio > DRIFT_THRESHOLD else ""
                    print(f"    {s}: {post_consolid_ppls[s]:.2f} / {best_ppls[s]:.2f} = {ratio:.2f}x{drift_flag}")

            drifted = verify_drift(post_consolid_ppls, best_ppls, completed_stages[:-1])
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}")
                for s, info in drifted.items():
                    print(f"    {s}: {info['current']:.2f} / {info['best']:.2f} = {info['ratio']:.2f}x")

                still_drifted = drifted
                for step in range(MAX_REPAIR_STEPS):
                    repair_toward_snapshot(model, neo_snapshot)
                    repair_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_stages[:-1])
                    if not still_drifted:
                        print(f"  [AVR] Converged at step {step+1}")
                        break
                if still_drifted:
                    print(f"  [AVR] Max steps ({MAX_REPAIR_STEPS}) reached, drift remains")
                neo_state = get_lora_state(model)
                total_repairs += step + 1
            else:
                print(f"  [AVR] No drift")

        # Update best PPLs
        final_ppls = eval_domain_ppls(model, tokenizer, stages, i+1)
        for s, p in final_ppls.items():
            if s not in best_ppls or p < best_ppls[s]:
                best_ppls[s] = p

        result = eval_all(model, tokenizer, mmlu_pairs, stages, i+1, f"after stage {i+1} ({name})")
        eval_results.append({"stage": name, "result": result})

        # Checkpoint
        checkpoint = {"method": "twostream_avr", "data_seed": DATA_SEED,
                      "completed_stages": i+1, "results_so_far": eval_results,
                      "total_repairs_so_far": total_repairs}
        with open(OUTPUT_DIR / f"checkpoint_twostream_d{DATA_SEED}_s{i+1}.json", "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        print(f"  [checkpoint] saved after stage {i+1}")

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    print(f"\n  [twostream+AVR] Total repair steps: {total_repairs}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return eval_results, total_repairs

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("SHIPPING TEST 2: Two-Stream + AVR on post-training pipeline")
    print(f"Model: {MODEL_ID}")
    print(f"Model seed: {MODEL_SEED} | Data seed: {DATA_SEED}")
    print(f"Stages: UltraChat → Medical → Code → Finance")
    print(f"Two-Stream: hippocampus (fresh per task) + neocortex (persistent)")
    print(f"Consolidation: KL distillation, lr={CONSOLIDATION_LR}")
    print(f"AVR: threshold={DRIFT_THRESHOLD}, α={REPAIR_ALPHA}, max_steps={MAX_REPAIR_STEPS}")
    print("=" * 70)

    # Load tokenizer
    print("\n  Loading tokenizer for data prep...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    stages = load_pipeline_data(tokenizer)
    print("\n  Loading MMLU eval set...")
    mmlu_pairs, mmlu_subjects = load_mmlu_eval(n_subjects=20, n_per_subject=5)
    print(f"    MMLU: {len(mmlu_pairs)} questions across {len(mmlu_subjects)} subjects")

    # Run two-stream + AVR
    twostream_results, total_repairs = run_twostream_avr_pipeline(stages, mmlu_pairs)

    # === RESULTS TABLE ===
    print(f"\n{'='*70}")
    print("SHIPPING COMPARISON — UPDATED TABLE")
    print(f"{'='*70}")

    domain_names = [s[0] for s in stages]
    print(f"\n{'Method':<18} {'MMLU':<8} " + " ".join(f"{n[:10]:<11}" for n in domain_names))
    print("-" * (18 + 8 + 12 * len(domain_names)))

    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "N/A"
        return f"{v:.1f}"

    # Previous results (from shipping_comparison.py run with DATA_SEED=42)
    print(f"{'Base':<18} {'0.370':<8} " + " ".join(f"{'—':<11}" for _ in domain_names) + "  (from prior run)")
    print(f"{'Naive SFT':<18} {'0.230':<8} " + " ".join(f"{'—':<11}" for _ in domain_names) + "  (from prior run)")
    print(f"{'AVR-alone':<18} {'0.290':<8} " + " ".join(f"{'—':<11}" for _ in domain_names) + "  (from prior run)")

    # This run
    ts_final = twostream_results[-1]["result"]
    ts_ppls = ts_final["domain_ppls"]
    row = f"{'TwoStream+AVR':<18} {ts_final['mmlu']:<8.3f} " + " ".join(f"{fmt(ts_ppls.get(n, float('nan'))):<11}" for n in domain_names)
    print(row)

    # Trajectory
    print(f"\n{'='*70}")
    print("TRAJECTORY (MMLU after each stage)")
    print(f"{'='*70}")
    print(f"\n{'Stage':<15} {'Base':<10} {'Naive':<10} {'AVR':<10} {'TwoStream':<12}")
    print("-" * 57)
    print(f"{'(base)':<15} {'0.370':<10} {'0.370':<10} {'0.370':<10} {'0.370':<12}")
    base_mmlu = 0.370
    naive_mmlu_trajectory = [0.350, 0.350, 0.270, 0.230]
    avr_mmlu_trajectory = [0.350, 0.290, 0.280, 0.290]
    for i, res in enumerate(twostream_results):
        ts_mmlu = res["result"]["mmlu"]
        stage = res["stage"]
        n_mmlu = naive_mmlu_trajectory[i] if i < len(naive_mmlu_trajectory) else "—"
        a_mmlu = avr_mmlu_trajectory[i] if i < len(avr_mmlu_trajectory) else "—"
        print(f"{stage:<15} {'—':<10} {n_mmlu:<10} {a_mmlu:<10} {ts_mmlu:<12.3f}")

    print(f"\n  Two-Stream + AVR total repair steps: {total_repairs}")

    # Save
    results = {
        "method": "twostream_avr",
        "data_seed": DATA_SEED,
        "model_seed": MODEL_SEED,
        "results": twostream_results,
        "total_repairs": total_repairs,
        "config": {
            "model": MODEL_ID, "lora_rank": LORA_RANK,
            "consolidation_lr": CONSOLIDATION_LR,
            "max_repair_steps": MAX_REPAIR_STEPS,
            "drift_threshold": DRIFT_THRESHOLD, "repair_alpha": REPAIR_ALPHA,
        },
        "prior_run_comparison": {
            "base_mmlu": 0.370, "naive_mmlu": 0.230,
            "avr_alone_mmlu": 0.290,
        }
    }
    out_name = f"shipping_twostream_d{DATA_SEED}.json"
    with open(OUTPUT_DIR / out_name, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {OUTPUT_DIR}/{out_name}")

if __name__ == "__main__":
    main()
