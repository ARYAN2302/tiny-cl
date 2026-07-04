"""
V24: AVR with gradient-based repair (vs v23's blind interpolation)

The ONLY change from v23: replace repair_toward_snapshot() with
gradient_repair(). Instead of blindly interpolating weights toward
the snapshot, we run a few gradient steps on a mixed objective:

    loss = loss_old(probe) + lambda * loss_new(current_task)

The gradient finds directions that recover old tasks with minimal
new-task damage. Weight interpolation can't do this — it's a fixed
geometric operation that doesn't know the loss landscape.

Same AVR loop: train → verify → repair. Only the repair mechanism changes.
Same config: LFM2.5-350M, LoRA r=32, TRACE 5000, threshold 1.15.

USAGE: Copy-paste into one Kaggle cell. ~2h on T4.
Change SEED below for multi-seed runs.
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.40", "peft>=0.10", "datasets>=2.14",
                "gdown>=4.7", "numpy"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y",
                "torchao"], check=False, capture_output=True)

import os, json, time, random, math, gc, copy, re
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
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("/home/z/my-project/download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_TASKS = ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"]
TRACE_VARIANT = "LLM-CL-Benchmark_5000"
TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"

LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["in_proj", "out_proj"]

TRAIN_LR = 2e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 3
BATCH_SIZE = 8
CONTEXT_LENGTH = 512

DRIFT_THRESHOLD = 1.15
MAX_REPAIR_STEPS = 100

# Gradient repair config
REPAIR_LR = 1e-4          # lower than training LR — surgical, not full retrain
REPAIR_LAMBDA = 0.5       # weight on new-task loss (0.5 = balance old and new)
REPAIR_BATCH_SIZE = 4     # small batches for repair
REPAIR_PROBE_SIZE = 50    # samples per drifted task per repair step

BENCH_MAX_NEW_TOKENS = 20
SEED = 42  # ← change to 123 or 7 for other seeds

CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# TRACE DATA
# ============================================================================

def download_trace_data():
    trace_dir = OUTPUT_DIR / "trace_data"
    if trace_dir.exists() and any(trace_dir.iterdir()):
        return trace_dir
    print("  Downloading TRACE...")
    import gdown
    zip_path = OUTPUT_DIR / "trace_benchmark.zip"
    gdown.download(id=TRACE_GDRIVE_ID, output=str(zip_path), quiet=False)
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(OUTPUT_DIR)
    for d in OUTPUT_DIR.rglob(f"*{TRACE_VARIANT}*"):
        if d.is_dir():
            trace_dir = d
            break
    print(f"  TRACE at: {trace_dir}")
    return trace_dir

def load_trace_task(trace_dir, task_name):
    task_dir = trace_dir / task_name
    with open(task_dir / "train.json") as f:
        train_data = json.load(f)
    with open(task_dir / "test.json") as f:
        test_data = json.load(f)
    train_pairs = [(ex["prompt"], ex["answer"]) for ex in train_data]
    test_pairs = [(ex["prompt"], ex["answer"]) for ex in test_data]
    print(f"    {task_name}: {len(train_pairs)} train, {len(test_pairs)} test")
    return train_pairs, test_pairs

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

# ============================================================================
# SNAPSHOT + DRIFT DETECTION (same as v23)
# ============================================================================

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def verify_drift(current_ppls, best_ppls, completed_tasks, threshold=DRIFT_THRESHOLD):
    drifted = {}
    for task in completed_tasks:
        if task not in current_ppls or task not in best_ppls:
            continue
        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0
        if ratio > threshold:
            drifted[task] = {"current_ppl": current_ppls[task], "best_ppl": best_ppls[task], "ratio": ratio}
    return drifted

# ============================================================================
# GRADIENT REPAIR (the new part — replaces v23's repair_toward_snapshot)
# ============================================================================

def build_token_stream(tokenizer, pairs):
    all_tokens = []
    for prompt, answer in pairs:
        text = prompt + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

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

def compute_loss_on_pairs(model, tokenizer, pairs, max_samples=REPAIR_PROBE_SIZE):
    """Compute average loss on a set of (prompt, answer) pairs."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for prompt, answer in pairs[:max_samples]:
            text = prompt + " " + answer + tokenizer.eos_token
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return total_loss / max(total_tokens, 1)

def gradient_repair(model, tokenizer, drifted_tasks, train_data, current_task,
                    current_task_pairs, snapshot_state, max_steps=MAX_REPAIR_STEPS):
    """Gradient-based repair: optimize old-task probe + new-task loss simultaneously.

    Unlike v23's blind interpolation (theta = 0.9*theta + 0.1*snapshot), this
    runs actual gradient steps on a mixed objective. The gradient finds
    directions that recover old tasks with minimal new-task damage.

    Args:
        model: the LoRA-wrapped model
        tokenizer: HF tokenizer
        drifted_tasks: list of task names that drifted
        train_data: {task_name: [(prompt, answer), ...]} for all tasks
        current_task: name of the task just trained
        current_task_pairs: training pairs for the current task
        snapshot_state: the pre-training LoRA snapshot (UNUSED in gradient repair,
                       kept for interface compatibility)

    Returns:
        number of repair steps taken
    """
    print(f"  [GRADIENT-REPAIR] Drifted: {drifted_tasks}")
    print(f"    lambda={REPAIR_LAMBDA} (new-task weight), lr={REPAIR_LR}")

    # Collect probe data: 50 samples from each drifted old task
    old_probe_pairs = []
    for task in drifted_tasks:
        old_probe_pairs.extend(train_data[task][:REPAIR_PROBE_SIZE])
    # Sample from current task to maintain new-task performance
    if len(current_task_pairs) > REPAIR_PROBE_SIZE:
        new_sample_pairs = random.sample(current_task_pairs, REPAIR_PROBE_SIZE)
    else:
        new_sample_pairs = current_task_pairs

    # Set up optimizer — only LoRA params, lower LR than training
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=REPAIR_LR, weight_decay=TRAIN_WD)

    # Pre-tokenize probe data into batches to avoid OOM
    # Process in small batches instead of one-at-a-time
    def make_batch(pairs, batch_size=REPAIR_BATCH_SIZE):
        """Tokenize pairs and return as batches of (input_ids, labels)."""
        all_input_ids = []
        for prompt, answer in pairs:
            text = prompt + " " + answer + tokenizer.eos_token
            ids = tokenizer.encode(text, add_special_tokens=False)[:CONTEXT_LENGTH]
            if len(ids) < 10: continue
            # Pad to CONTEXT_LENGTH
            ids = ids + [tokenizer.pad_token_id] * (CONTEXT_LENGTH - len(ids))
            all_input_ids.append(ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        batches = []
        for i in range(0, len(all_input_ids), batch_size):
            batch = all_input_ids[i:i+batch_size].to(DEVICE)
            batches.append(batch)
        return batches

    old_batches = make_batch(old_probe_pairs, REPAIR_BATCH_SIZE)
    new_batches = make_batch(new_sample_pairs, REPAIR_BATCH_SIZE)

    steps_taken = 0
    for step in range(max_steps):
        model.train()

        # --- Old-task loss (one random batch per step) ---
        old_batch = random.choice(old_batches)
        old_out = model(input_ids=old_batch, labels=old_batch)
        old_loss = old_out.loss

        # --- New-task loss (one random batch per step) ---
        new_batch = random.choice(new_batches)
        new_out = model(input_ids=new_batch, labels=new_batch)
        new_loss = new_out.loss

        # --- Combined loss ---
        loss = old_loss + REPAIR_LAMBDA * new_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
        opt.step()
        steps_taken += 1

        # Free intermediate tensors
        del old_out, new_out, loss
        if steps_taken % 3 == 0:
            torch.cuda.empty_cache()

        if step % 5 == 0 or step < 5:
            print(f"    [GRADIENT-REPAIR] step {step+1} | "
                  f"old_loss={old_loss.item():.4f} | new_loss={new_loss.item():.4f} | "
                  f"total={(old_loss + REPAIR_LAMBDA * new_loss).item():.4f}", flush=True)

        # Check convergence every 5 steps (PPL eval is expensive)
        if (step + 1) % 5 == 0:
            # Re-verify old tasks
            current_ppls = {}
            for task in drifted_tasks:
                current_ppls[task] = compute_ppl(model, tokenizer, train_data[task], 50)
            # Check if all drifted tasks are back below threshold
            all_fixed = True
            for task in drifted_tasks:
                # We need best_ppls to compare — pass it in or recompute
                # For now, just check if PPL stopped improving
                pass
            # Simpler convergence: check if old_loss is below a fraction of initial
            # We'll just run a fixed number and check at the end
            # (Full convergence check adds complexity — keep it simple for first test)

        if steps_taken >= 15:  # cap at 15 for first test — matches v23's effective range
            break

    print(f"  [GRADIENT-REPAIR] Done: {steps_taken} steps")
    return steps_taken

# ============================================================================
# TRAINING (same as v23)
# ============================================================================

def train_on_pairs(model, tokenizer, pairs, epochs=TASK_EPOCHS):
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
    return gs, tl

# ============================================================================
# EVALUATION (same as v23)
# ============================================================================

def compute_ppl(model, tokenizer, pairs, max_samples=200):
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

def eval_all_ppls(model, tokenizer, train_data, task_order, trained_so_far, max_samples=200):
    ppls = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far: break
        ppls[task] = compute_ppl(model, tokenizer, train_data[task], max_samples)
    return ppls

def generate(model, tokenizer, prompt, max_new_tokens=BENCH_MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def score_answer(response, gold):
    response = response.strip(); gold = gold.strip()
    if gold in ["A", "B", "C", "D", "E"]:
        response_upper = response.upper()[:5]
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in response_upper:
                return 1.0 if letter == gold else 0.0
        return 0.0
    if re.match(r'^[\d.-]+', gold):
        numbers = re.findall(r'[\d.]+', response)
        if numbers: return 1.0 if numbers[-1] == gold else 0.0
        return 0.0
    def norm(s):
        s = s.lower().strip(); s = re.sub(r'[^\w\s.-]', ' ', s); return ' '.join(s.split())
    return 1.0 if norm(response) == norm(gold) else 0.0

def evaluate_task_accuracy(model, tokenizer, test_pairs, task_name, max_questions=200):
    print(f"    Eval {task_name} ({min(len(test_pairs), max_questions)} Qs)...", flush=True)
    correct = 0; total = min(len(test_pairs), max_questions)
    for i in range(total):
        prompt, gold = test_pairs[i]
        response = generate(model, tokenizer, prompt)
        if score_answer(response, gold): correct += 1
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f}")
    return acc

def compute_metrics(R, task_order):
    T = len(task_order)
    ACC = np.mean([R[T-1][j] for j in range(T)])
    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]
    BWT = np.mean(bwt_values) if bwt_values else 0.0
    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]
    FF = np.mean(ff_values) if ff_values else 0.0
    return {"ACC": float(ACC), "BWT": float(BWT), "FF": float(FF)}

# ============================================================================
# RUN AVR WITH GRADIENT REPAIR
# ============================================================================

def run_avr(train_data, test_data, task_order):
    print(f"\n{'#'*70}")
    print(f"# AVR with GRADIENT REPAIR (seed={SEED})")
    print(f"# repair: gradient steps on old_probe + lambda*new_task")
    print(f"# lambda={REPAIR_LAMBDA}, lr={REPAIR_LR}, max_steps=15")
    print(f"{'#'*70}")

    model, tokenizer = create_model()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]

    merged_snapshot = None
    best_ppls = {}
    completed_tasks = []
    total_repair_steps = 0
    repair_log = []
    current_task_pairs = None  # stored for gradient repair

    for task_idx, task in enumerate(task_order):
        task_num = task_idx + 1
        print(f"\n{'='*60}")
        print(f"  Task {task_num}/{T}: {task}")
        print(f"{'='*60}", flush=True)

        train_pairs = train_data[task]
        current_task_pairs = train_pairs  # store for repair

        gs, tl = train_on_pairs(model, tokenizer, train_pairs)
        current_state = get_lora_state(model)

        post_train_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        if task not in best_ppls:
            best_ppls[task] = post_train_ppls[task]
        completed_tasks.append(task)

        print(f"  Post-train PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_train_ppls.items()))

        phase_repair_steps = 0
        if task_num > 1 and merged_snapshot is not None:
            drifted = verify_drift(post_train_ppls, best_ppls, completed_tasks[:-1])
            if drifted:
                print(f"  [AVR] DRIFT DETECTED on {list(drifted.keys())}:")
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x")

                # === GRADIENT REPAIR (new) ===
                phase_repair_steps = gradient_repair(
                    model, tokenizer,
                    drifted_tasks=list(drifted.keys()),
                    train_data=train_data,
                    current_task=task,
                    current_task_pairs=current_task_pairs,
                    snapshot_state=merged_snapshot,  # unused but kept for interface
                )

                # Re-eval PPLs after repair
                post_repair_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
                print(f"  Post-repair PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_repair_ppls.items()))
            else:
                print(f"  [AVR] No drift — repair not needed")

        total_repair_steps += phase_repair_steps
        repair_log.append({"task": task, "repair_steps": phase_repair_steps})

        # Update best PPLs
        final_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        for dpk, dppl in final_ppls.items():
            if dpk not in best_ppls or dppl < best_ppls[dpk]:
                best_ppls[dpk] = dppl

        merged_snapshot = get_lora_state(model)

        print(f"\n  Evaluating accuracy on all tasks...", flush=True)
        for j in range(task_idx + 1):
            R[task_idx][j] = evaluate_task_accuracy(model, tokenizer, test_data[task_order[j]], task_order[j])

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    metrics = compute_metrics(R, task_order)
    print(f"\n  [AVR] Total repair steps: {total_repair_steps}")
    print(f"  [AVR] Repair log: {repair_log}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics, total_repair_steps, repair_log

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"V24: AVR with GRADIENT REPAIR on TRACE")
    print(f"Seed: {SEED} | Tasks: {TRACE_TASKS}")
    print(f"Repair: gradient steps, lambda={REPAIR_LAMBDA}, lr={REPAIR_LR}")
    print("=" * 70)

    print("\nDownloading TRACE data...")
    trace_dir = download_trace_data()

    print("\nLoading task data...")
    train_data, test_data = {}, {}
    for task in TRACE_TASKS:
        train_data[task], test_data[task] = load_trace_task(trace_dir, task)

    avr_R, avr_metrics, total_repairs, repair_log = run_avr(train_data, test_data, TRACE_TASKS)

    print(f"\n{'='*70}")
    print("THE VERDICT: AVR with GRADIENT REPAIR")
    print(f"{'='*70}")

    print(f"\n{'Method':<40} {'ACC':<10} {'BWT':<10} {'FF':<10} {'Repairs':<10}")
    print("-" * 80)
    print(f"{'Naive (v18)':<40} {'0.379':<10} {'-0.130':<10} {'0.130':<10} {'—':<10}")
    print(f"{'SLAO (v18)':<40} {'0.397':<10} {'-0.062':<10} {'0.062':<10} {'—':<10}")
    print(f"{'AVR interp (v23, s42)':<40} {'0.374':<10} {'-0.023':<10} {'0.038':<10} {'24':<10}")
    print(f"{'AVR interp (s123)':<40} {'0.261':<10} {'-0.002':<10} {'0.005':<10} {'205':<10}")
    print(f"{'AVR GRADIENT (this run)':<40} {avr_metrics['ACC']:<10.3f} {avr_metrics['BWT']:<10.3f} {avr_metrics['FF']:<10.3f} {total_repairs:<10}")

    print(f"\n  Repair steps per task:")
    for entry in repair_log:
        print(f"    {entry['task']}: {entry['repair_steps']} repair steps")

    print(f"\n  R MATRIX (AVR gradient repair):")
    header = "  After\\Test  " + "  ".join(f"{t[:8]:<10}" for t in TRACE_TASKS)
    print(header)
    for i in range(len(TRACE_TASKS)):
        print(f"  {TRACE_TASKS[i][:8]:<10} " + "  ".join(f"{avr_R[i][j]:<10.3f}" for j in range(len(TRACE_TASKS))))

    results = {
        "seed": SEED, "tasks": TRACE_TASKS,
        "method": "avr_gradient_repair",
        "metrics": avr_metrics, "R": avr_R,
        "total_repair_steps": total_repairs, "repair_log": repair_log,
        "config": {"lambda": REPAIR_LAMBDA, "lr": REPAIR_LR, "max_steps": 15},
    }
    with open(OUTPUT_DIR / f"v24_gradient_seed{SEED}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults: {OUTPUT_DIR}/v24_gradient_seed{SEED}_results.json")

if __name__ == "__main__":
    main()
