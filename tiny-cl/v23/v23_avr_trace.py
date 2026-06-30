"""
V23: AVR (standalone) on TRACE — PPL-ratio verify + closed-form repair
======================================================================

AVR alone. No SLAO. Just:
1. Train on task
2. Check PPL drift on previous tasks
3. If drifted: repair toward snapshot
4. Next task

This is AVR as it ran in the dashboard: naive vs anchor, standalone.
Reuses v18 naive numbers. Only runs AVR.

USAGE: !python v23_avr_trace.py
Runtime: ~2 hours on T4
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
REPAIR_ALPHA = 0.1       # pull strength per repair step
MAX_REPAIR_STEPS = 100   # no practical cap — repair until drift is fixed

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
from peft.tuners.lora.layer import LoraLayer

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
# SLAO CORE (from v13/v18)
# ============================================================================

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

def extract_orthogonal_A(model):
    ortho_A = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        A = module.lora_A["default"].weight.data.float()
        Q, R = torch.linalg.qr(A.T.contiguous())
        signs = torch.sign(torch.diag(R))
        Q = Q * signs.unsqueeze(0)
        ortho_A[name] = Q.T
    return ortho_A

def initialize_slao(model, ortho_A, prev_ft_B):
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if "default" not in module.lora_A: continue
        if name in ortho_A:
            module.lora_A["default"].weight.data.copy_(
                ortho_A[name].to(DEVICE).to(module.lora_A["default"].weight.data.dtype))
        B_key = f"{name}.lora_B.default.weight"
        if B_key in prev_ft_B:
            module.lora_B["default"].weight.data.copy_(
                prev_ft_B[B_key].to(DEVICE).to(module.lora_B["default"].weight.data.dtype))

def slao_merge_B(merged_state, ft_state, task_num):
    lam = 1.0 / math.sqrt(task_num)
    new_merged = {}
    for key in ft_state:
        ft_val = ft_state[key]
        if key in merged_state:
            if "lora_A" in key:
                new_merged[key] = ft_val.cpu().clone()
            elif "lora_B" in key:
                old_val = merged_state[key]
                new_merged[key] = (old_val + lam * (ft_val - old_val)).cpu().clone()
            else:
                new_merged[key] = ft_val.cpu().clone()
        else:
            new_merged[key] = ft_val.cpu().clone()
    return new_merged

# ============================================================================
# AVR CORE (from v11 — PPL-ratio verify + closed-form repair)
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

def repair_toward_snapshot(model, snapshot_state, alpha=REPAIR_ALPHA):
    """Pull LoRA weights toward snapshot: θ = (1-α)θ + α·θ_snapshot
    Closed-form interpolation — no optimizer, no gradients, just weight math.
    This is the v11 mechanism.
    """
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(DEVICE)
            p.data.copy_((1 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj

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
# RUN SLAO+AVR (v11 style — PPL-ratio verify + closed-form repair)
# ============================================================================

def run_avr(train_data, test_data, task_order):
    """AVR standalone — no SLAO. Just:
    1. Train on task (plain SFT)
    2. Check PPL drift on previous tasks
    3. If drifted: repair toward snapshot (closed-form interpolation)
    4. Snapshot current state for next phase
    """
    print(f"\n{'#'*70}")
    print(f"# AVR (standalone — PPL-ratio verify + closed-form repair)")
    print(f"# drift_threshold={DRIFT_THRESHOLD} | repair_alpha={REPAIR_ALPHA}")
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

        # Plain SFT — no SLAO init, no orthogonal projection
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

        if task_num > 1 and merged_snapshot is not None:
            # Verify: check PPL drift on ALL previous tasks
            drifted = verify_drift(post_train_ppls, best_ppls, completed_tasks[:-1])

            if drifted:
                print(f"  [AVR] DRIFT DETECTED on {list(drifted.keys())}:")
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x")

                # Repair loop: closed-form interpolation toward snapshot
                still_drifted = drifted
                for step in range(MAX_REPAIR_STEPS):
                    n_adj = repair_toward_snapshot(model, merged_snapshot)
                    phase_repair_steps += 1

                    # Re-evaluate PPL after repair
                    repair_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_tasks[:-1])

                    print(f"    [AVR] Repair step {step+1}: {n_adj} params adjusted, "
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
        repair_log.append({"task": task, "repair_steps": phase_repair_steps})

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
    print(f"  [AVR] Repair log: {repair_log}")

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return R, metrics, total_repair_steps, repair_log

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V23: Your AVR (v11 style) on TRACE")
    print(f"Seed: {SEED} | Tasks: {TRACE_TASKS}")
    print(f"AVR: PPL-ratio verify (threshold={DRIFT_THRESHOLD}) + closed-form repair (alpha={REPAIR_ALPHA})")
    print("=" * 70)

    # Download TRACE
    print("\nDownloading TRACE data...")
    trace_dir = download_trace_data()

    # Load tasks
    print("\nLoading task data...")
    train_data, test_data = {}, {}
    for task in TRACE_TASKS:
        train_data[task], test_data[task] = load_trace_task(trace_dir, task)

    # v18 results (reuse)
    naive_v18 = {"ACC": 0.379, "BWT": -0.130, "FF": 0.130}
    slao_v18 = {"ACC": 0.397, "BWT": -0.062, "FF": 0.062}
    avr_v19_broken = {"ACC": 0.405, "BWT": -0.082, "FF": 0.082}

    # --- Run AVR (standalone) ---
    avr_R, avr_metrics, total_repairs, repair_log = run_avr(train_data, test_data, TRACE_TASKS)

    # --- VERDICT ---
    print(f"\n{'='*70}")
    print("THE VERDICT: Your AVR (v11 style) on TRACE")
    print(f"{'='*70}")

    print(f"\n{'Method':<35} {'ACC':<10} {'BWT':<10} {'FF':<10} {'Repairs':<10}")
    print("-" * 75)
    print(f"{'Naive (v18)':<35} {naive_v18['ACC']:<10.3f} {naive_v18['BWT']:<10.3f} {naive_v18['FF']:<10.3f} {'—':<10}")
    print(f"{'SLAO (v18, published method)':<35} {slao_v18['ACC']:<10.3f} {slao_v18['BWT']:<10.3f} {slao_v18['FF']:<10.3f} {'—':<10}")
    print(f"{'AVR broken (v19, hidden-state)':<35} {avr_v19_broken['ACC']:<10.3f} {avr_v19_broken['BWT']:<10.3f} {avr_v19_broken['FF']:<10.3f} {'0':<10}")
    print(f"{'AVR (PPL-ratio + closed-form)':<35} {avr_metrics['ACC']:<10.3f} {avr_metrics['BWT']:<10.3f} {avr_metrics['FF']:<10.3f} {total_repairs:<10}")

    # Delta from naive and SLAO
    d_naive_acc = avr_metrics["ACC"] - naive_v18["ACC"]
    d_naive_bwt = avr_metrics["BWT"] - naive_v18["BWT"]
    d_slao_acc = avr_metrics["ACC"] - slao_v18["ACC"]
    d_slao_bwt = avr_metrics["BWT"] - slao_v18["BWT"]
    print(f"\n{'Delta from naive':<35} {d_naive_acc:<+10.3f} {d_naive_bwt:<+10.3f}")
    print(f"{'Delta from SLAO':<35} {d_slao_acc:<+10.3f} {d_slao_bwt:<+10.3f}")

    # Repair summary
    print(f"\n  Repair steps per task:")
    for entry in repair_log:
        print(f"    {entry['task']}: {entry['repair_steps']} repair steps")

    print(f"\n{'='*70}")
    if avr_metrics["ACC"] > naive_v18["ACC"] and total_repairs > 0:
        print(f"  AVR BEATS NAIVE — repairs fired {total_repairs} times, ACC {avr_metrics['ACC']:.3f} vs {naive_v18['ACC']:.3f}")
    elif total_repairs > 0:
        print(f"  AVR FIRED — {total_repairs} repairs, ACC {avr_metrics['ACC']:.3f} vs naive {naive_v18['ACC']:.3f}")
    else:
        print(f"  AVR DID NOT FIRE — no PPL drift above {DRIFT_THRESHOLD}x threshold")

    # R matrix
    print(f"\n  R MATRIX (AVR standalone):")
    header = "After\\Test  " + "  ".join(f"{t[:8]:<10}" for t in TRACE_TASKS)
    print(f"  {header}")
    for i in range(len(TRACE_TASKS)):
        row = f"  {TRACE_TASKS[i][:8]:<10} " + "  ".join(f"{avr_R[i][j]:<10.3f}" for j in range(len(TRACE_TASKS)))
        print(row)

    # Save
    results = {
        "seed": SEED, "tasks": TRACE_TASKS,
        "avr_standalone": {"metrics": avr_metrics, "R": avr_R,
                          "total_repair_steps": total_repairs, "repair_log": repair_log},
        "comparison": {"naive_v18": naive_v18, "slao_v18": slao_v18, "avr_v19_broken": avr_v19_broken},
    }
    with open(OUTPUT_DIR / "v23_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/v23_results.json")

if __name__ == "__main__":
    main()
