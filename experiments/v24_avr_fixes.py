"""
V24: AVR + accuracy fixes — adaptive α + selective repair
========================================================

Same as v23 (AVR standalone on TRACE, PPL-ratio verify + closed-form repair)
but with two fixes baked in to address the plasticity tax.

The plasticity tax: v23 pulls ALL LoRA weights toward the snapshot during
repair, which preserves old tasks but undoes new-task learning.

FIX A — ADAPTIVE α
  Scale repair strength by drift severity instead of fixed α=0.1.
  α = clip(0.1 + 0.1·(ratio − 1.15), 0.05, 0.20)
  Mild drift (1.16×) → α=0.10 (gentle); severe drift (2.0×) → α=0.19 (hard).
  Also stop repairing once drift drops below 1.10 (was 1.15 — was over-repairing).

FIX B — SELECTIVE REPAIR
  Don't pull all LoRA weights toward snapshot. Only pull the modules that
  actually contribute to drift on the affected tasks.
  attribution(m) = |grad_ppl_drifted| × |Δθ_m_since_snapshot|
  Repair only the top 30% highest-attribution modules. The other 70%
  (new-task-specific modules) keep their current weights.

Both fixes are always on in this file. Run it, compare to v23's output.

USAGE: !python v24_avr_fixes.py
Runtime: ~2 hours on T4 (slightly slower than v23 due to attribution grads)
"""

import subprocess, sys, os, json, time, random, math, gc, re, copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

# AVR config (from v11 — PPL-ratio, NOT hidden-state)
DRIFT_THRESHOLD = 1.15   # fire repair if PPL > 1.15x best
REPAIR_ALPHA = 0.1       # base pull strength (used as floor in adaptive α)
MAX_REPAIR_STEPS = 100   # no practical cap — repair until drift is fixed

# Fix A — Adaptive α
ADAPTIVE_ALPHA_MIN = 0.05   # floor for adaptive α
ADAPTIVE_ALPHA_MAX = 0.20   # ceiling for adaptive α
CONVERGE_BELOW = 1.10       # stop repairing once drift drops below this ratio

# Fix B — Selective repair
TOP_K_PCT = 0.30             # repair the top 30% highest-attribution LoRA modules
ATTRIBUTION_BATCH = 16       # samples per drifted task for gradient attribution

BENCH_MAX_NEW_TOKENS = 20
SEED = 42

CONV_LAYER_IDS = [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]
ATTN_LAYER_IDS = [2, 5, 8, 10, 12, 14]

# ============================================================================
# DEPS
# ============================================================================

def _ensure_deps():
    missing = []
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade",
                                   "transformers>=5.0.0", "packaging"])
    except ImportError:
        missing.extend(["transformers>=5.0.0", "packaging"])
    for pkg in ["peft", "numpy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    try: __import__("gdown")
    except ImportError: missing.append("gdown")
    if missing:
        print(f"Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)

_ensure_deps()

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

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
# LoRA STATE UTILITIES
# ============================================================================

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

# ============================================================================
# AVR CORE — PPL-ratio verify + closed-form repair (with Fix A + Fix B)
# ============================================================================

def verify_drift(current_ppls, best_ppls, completed_tasks, threshold=DRIFT_THRESHOLD):
    """Check if any previous task has PPL > threshold × best PPL.
    This is the v11 mechanism: PPL-ratio gate, NOT hidden-state MSE.
    """
    drifted = {}
    for task in completed_tasks:
        if task not in current_ppls or task not in best_ppls:
            continue
        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0
        if ratio > threshold:
            drifted[task] = {
                "current_ppl": current_ppls[task],
                "best_ppl": best_ppls[task],
                "ratio": ratio,
            }
    return drifted

def adaptive_alpha(ratio):
    """Fix A: scale repair strength by drift severity.
    Mild drift (ratio near 1.15) → α ≈ 0.10 (gentle)
    Severe drift (ratio > 1.5)   → α ≈ 0.20 (hard pull)
    """
    return float(np.clip(REPAIR_ALPHA + 0.1 * (ratio - DRIFT_THRESHOLD),
                         ADAPTIVE_ALPHA_MIN, ADAPTIVE_ALPHA_MAX))

def compute_drift_attribution(model, tokenizer, drifted_tasks, train_data, snapshot_state):
    """Fix B: for each LoRA module, compute how much its update since snapshot
    contributes to drift on the drifted tasks.

    attribution(m) = |∂PPL_drifted / ∂θ_m| × |θ_m − θ_snapshot_m|

    High attribution = this module's recent update is causing the drift.
    Low attribution  = this module is either irrelevant to drifted tasks,
                       or hasn't changed much (likely new-task-specific).
    """
    model.eval()
    # Zero out all grads
    for n, p in model.named_parameters():
        if p.grad is not None: p.grad.zero_()

    n_samples = ATTRIBUTION_BATCH
    n_tasks = len(drifted_tasks)
    for task in drifted_tasks:
        pairs = train_data[task][:n_samples]
        for prompt, answer in pairs:
            text = prompt + " " + answer + tokenizer.eos_token
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            with torch.enable_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                n_tok = inputs["input_ids"].shape[1]
                (outputs.loss / (n_tasks * n_tok)).backward()

    attributions = {}
    for n, p in model.named_parameters():
        if "lora_" not in n or n not in snapshot_state:
            continue
        grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
        delta_norm = (p.data - snapshot_state[n].to(DEVICE)).norm().item()
        attributions[n] = grad_norm * delta_norm
        if p.grad is not None: p.grad.zero_()

    model.train()
    return attributions

def repair_toward_snapshot(model, snapshot_state, drifted, attributions=None):
    """Pull LoRA weights toward snapshot: θ = (1−α)·θ + α·θ_snapshot.

    Fix A: α is adaptive — scaled by the max drift ratio across drifted tasks.
    Fix B: only repair modules whose attribution is in the top-k%. Other modules
           keep their current (new-task-trained) weights.
    """
    # Fix A: use the max α across drifted tasks (most severe drives repair strength)
    alpha = max(adaptive_alpha(info["ratio"]) for info in drifted.values())

    # Fix B: determine which modules to repair
    if attributions is not None and len(attributions) > 0:
        threshold_val = float(np.percentile(
            list(attributions.values()),
            100 * (1 - TOP_K_PCT)
        ))
        modules_to_repair = {n for n, a in attributions.items() if a >= threshold_val}
    else:
        modules_to_repair = None  # None means "all"

    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" not in n or n not in snapshot_state:
            continue
        if modules_to_repair is not None and n not in modules_to_repair:
            continue  # Fix B: skip this module (preserve new-task learning)
        snap_val = snapshot_state[n].to(DEVICE)
        p.data.copy_((1 - alpha) * p.data + alpha * snap_val)
        n_adj += 1
    return n_adj, alpha

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

def build_training_stream(tokenizer, pairs):
    all_tokens = []
    for prompt, answer in pairs:
        text = prompt + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

def train_on_pairs(model, tokenizer, pairs, epochs=TASK_EPOCHS):
    token_ids = build_training_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    Training: {len(token_ids):,} tokens, {len(dataset)} chunks")

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
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
            if gs % 100 == 0: print(f"      step {gs} | loss={tl/gs:.4f}")
    return gs, tl

# ============================================================================
# EVALUATION
# ============================================================================

def compute_ppl(model, tokenizer, pairs, max_samples=200):
    """Compute perplexity on a set of (prompt, answer) pairs."""
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
    """Evaluate PPL on all tasks seen so far (for AVR drift detection)."""
    ppls = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far:
            break
        ppls[task] = compute_ppl(model, tokenizer, train_data[task], max_samples)
    return ppls

def generate(model, tokenizer, prompt, max_new_tokens=BENCH_MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

def score_answer(response, gold):
    response = response.strip()
    gold = gold.strip()
    if gold in ["A", "B", "C", "D", "E"]:
        response_upper = response.upper()[:5]
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in response_upper:
                return 1.0 if letter == gold else 0.0
        return 0.0
    if re.match(r'^[\d.-]+', gold):
        numbers = re.findall(r'[\d.]+', response)
        if numbers:
            return 1.0 if numbers[-1] == gold else 0.0
        return 0.0
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s.-]', ' ', s)
        return ' '.join(s.split())
    return 1.0 if norm(response) == norm(gold) else 0.0

def evaluate_task_accuracy(model, tokenizer, test_pairs, task_name, max_questions=200):
    """Evaluate accuracy (not PPL) on a task's test set."""
    print(f"    Eval {task_name} ({min(len(test_pairs), max_questions)} Qs)...")
    correct = 0
    total = min(len(test_pairs), max_questions)
    for i in range(total):
        prompt, gold = test_pairs[i]
        response = generate(model, tokenizer, prompt, max_new_tokens=BENCH_MAX_NEW_TOKENS)
        if score_answer(response, gold):
            correct += 1
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
# RUN AVR
# ============================================================================

def run_avr(train_data, test_data, task_order):
    """AVR standalone. Plain SFT + verify + repair.
    1. Train on task (plain SFT)
    2. Check PPL drift on previous tasks
    3. If drifted: repair toward snapshot (closed-form interpolation)
    4. Snapshot current state for next phase
    """
    print(f"\n{'#'*70}")
    print(f"# AVR (standalone) — PPL-ratio verify + closed-form repair")
    print(f"# drift_threshold={DRIFT_THRESHOLD} | converge_below={CONVERGE_BELOW} | base_alpha={REPAIR_ALPHA}")
    print(f"# Fix A (adaptive α)    = ON  (range [{ADAPTIVE_ALPHA_MIN}, {ADAPTIVE_ALPHA_MAX}])")
    print(f"# Fix B (selective)     = ON  (top {TOP_K_PCT*100:.0f}% by attribution)")
    print(f"{'#'*70}")

    model, tokenizer = create_model()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]

    merged_snapshot = None    # previous state for repair target
    best_ppls = {}            # best PPL seen for each task
    completed_tasks = []
    total_repair_steps = 0
    repair_log = []

    for task_idx, task in enumerate(task_order):
        task_num = task_idx + 1
        print(f"\n{'='*60}")
        print(f"  Task {task_num}/{T}: {task}")
        print(f"{'='*60}")

        train_pairs = train_data[task]

        # Plain SFT on this task
        gs, tl = train_on_pairs(model, tokenizer, train_pairs)

        # Get current LoRA state
        current_state = get_lora_state(model)

        # Post-train PPL (for AVR drift detection)
        post_train_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        if task not in best_ppls:
            best_ppls[task] = post_train_ppls[task]
        completed_tasks.append(task)

        print(f"  Post-train PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_train_ppls.items()))

        # ── AVR VERIFY + REPAIR ──
        phase_repair_steps = 0
        drifted = {}           # always defined — empty when no drift check runs
        phase_alphas = []      # always defined — empty when no repair runs

        if task_num > 1 and merged_snapshot is not None:
            # Verify: check PPL drift on ALL previous tasks
            drifted = verify_drift(post_train_ppls, best_ppls, completed_tasks[:-1])

            if drifted:
                print(f"  [AVR] DRIFT DETECTED on {list(drifted.keys())}:")
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x")

                # Repair loop: closed-form interpolation toward snapshot
                still_drifted = drifted
                phase_alphas = []
                for step in range(MAX_REPAIR_STEPS):
                    # Fix B: compute attribution once per repair step (changes as we repair)
                    attributions = compute_drift_attribution(
                        model, tokenizer, list(still_drifted.keys()),
                        train_data, merged_snapshot
                    )

                    n_adj, alpha_used = repair_toward_snapshot(
                        model, merged_snapshot, still_drifted, attributions
                    )
                    phase_repair_steps += 1
                    phase_alphas.append(alpha_used)

                    # Re-evaluate PPL after repair
                    repair_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
                    # Fix A: use CONVERGE_BELOW (1.10) as stop threshold, not DRIFT_THRESHOLD
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_tasks[:-1], threshold=CONVERGE_BELOW)

                    print(f"    [AVR] Repair step {step+1}: α={alpha_used:.3f}, {n_adj} params adjusted, "
                          f"still drifted: {list(still_drifted.keys()) if still_drifted else 'none'}")

                    if not still_drifted:
                        print(f"  [AVR] Repair converged at step {step+1}")
                        break

                if still_drifted:
                    print(f"  [AVR] Max repair steps ({MAX_REPAIR_STEPS}) reached, "
                          f"drift remains on {list(still_drifted.keys())}")

                # Update current state with repaired model
                current_state = get_lora_state(model)
            else:
                print(f"  [AVR] No drift — repair not needed")

        total_repair_steps += phase_repair_steps
        repair_log.append({"task": task, "repair_steps": phase_repair_steps, "alphas": phase_alphas})

        # Final PPL after repair (if any)
        final_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)

        # Update best PPLs
        for dpk, dppl in final_ppls.items():
            if dpk not in best_ppls or dppl < best_ppls[dpk]:
                best_ppls[dpk] = dppl

        # Snapshot current state for next phase's repair target
        merged_snapshot = copy.deepcopy(current_state)

        # Evaluate accuracy on all tasks seen so far (for R matrix)
        print(f"\n  Evaluating accuracy on all tasks...")
        for j in range(task_idx + 1):
            R[task_idx][j] = evaluate_task_accuracy(model, tokenizer, test_data[task_order[j]], task_order[j])

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    metrics = compute_metrics(R, task_order)

    print(f"\n  [AVR] Total repair steps: {total_repair_steps}")
    print(f"  [AVR] Repair log:")
    for entry in repair_log:
        a_str = f", alphas={[f'{a:.3f}' for a in entry['alphas']]}" if entry['alphas'] else ""
        print(f"    {entry['task']}: {entry['repair_steps']} repair steps{a_str}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return R, metrics, total_repair_steps, repair_log

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V24: AVR + accuracy fixes on TRACE")
    print(f"Seed: {SEED} | Tasks: {TRACE_TASKS}")
    print(f"AVR: PPL-ratio verify (threshold={DRIFT_THRESHOLD}, converge_below={CONVERGE_BELOW})")
    print(f"     + closed-form repair (base α={REPAIR_ALPHA}, adaptive [{ADAPTIVE_ALPHA_MIN}-{ADAPTIVE_ALPHA_MAX}])")
    print(f"Fix A (adaptive α)    = ON")
    print(f"Fix B (selective repair) = ON (top {TOP_K_PCT*100:.0f}% by attribution)")
    print("=" * 70)

    # Download TRACE
    print("\nDownloading TRACE data...")
    trace_dir = download_trace_data()

    # Load tasks
    print("\nLoading task data...")
    train_data, test_data = {}, {}
    for task in TRACE_TASKS:
        train_data[task], test_data[task] = load_trace_task(trace_dir, task)

    # --- Run AVR (with fixes) ---
    avr_R, avr_metrics, total_repairs, repair_log = run_avr(train_data, test_data, TRACE_TASKS)

    # --- VERDICT: only v23 vs v24 ---
    print(f"\n{'='*70}")
    print("v23 vs v24 (the two fixes)")
    print(f"{'='*70}")

    print(f"\n{'Method':<35} {'ACC':<10} {'BWT':<10} {'FF':<10} {'Repairs':<10}")
    print("-" * 75)
    print(f"{'v23 (no fixes)':<35} {0.374:<10.3f} {-0.023:<10.3f} {0.038:<10.3f} {24:<10}")
    print(f"{'v24 (adaptive α + selective)':<35} {avr_metrics['ACC']:<10.3f} {avr_metrics['BWT']:<10.3f} {avr_metrics['FF']:<10.3f} {total_repairs:<10}")

    # Delta from v23
    d_acc = avr_metrics["ACC"] - 0.374
    d_bwt = avr_metrics["BWT"] - (-0.023)
    d_ff = avr_metrics["FF"] - 0.038
    print(f"\n{'Delta (v24 − v23)':<35} {d_acc:<+10.3f} {d_bwt:<+10.3f} {d_ff:<+10.3f}")

    # R matrix
    print(f"\n  R MATRIX (v24):")
    header = "After\\Test  " + "  ".join(f"{t[:8]:<10}" for t in TRACE_TASKS)
    print(f"  {header}")
    for i in range(len(TRACE_TASKS)):
        row = f"  {TRACE_TASKS[i][:8]:<10} " + "  ".join(f"{avr_R[i][j]:<10.3f}" for j in range(len(TRACE_TASKS)))
        print(row)

    # Save — only v24's numbers, no comparisons
    results = {
        "seed": SEED,
        "tasks": TRACE_TASKS,
        "version": "v24",
        "fixes": {
            "adaptive_alpha": True,
            "selective_repair": True,
            "top_k_pct": TOP_K_PCT,
            "alpha_range": [ADAPTIVE_ALPHA_MIN, ADAPTIVE_ALPHA_MAX],
            "converge_below": CONVERGE_BELOW,
        },
        "metrics": avr_metrics,
        "R": avr_R,
        "total_repair_steps": total_repairs,
        "repair_log": repair_log,
    }
    with open(OUTPUT_DIR / "v24_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/v24_results.json")

if __name__ == "__main__":
    main()
