"""
V18: The Real Test — TRACE Benchmark with SLAO+MVA
===================================================

Tests the living model on the recognized continual learning benchmark.
One file, one process, one run.

WHAT IT DOES
------------
1. Download TRACE data (500 variant, 4 tasks that work on 350M)
2. Load LFM2.5-350M + LoRA
3. Evaluate baseline on all 4 tasks (our harness, our prompts)
4. Stream tasks sequentially:
   - Method A: NAIVE (sequential SFT, no protection)
   - Method B: SLAO+MVA (the living model mechanism)
   After each task, evaluate ALL tasks seen so far
5. Build the R matrix (score on task j after training task i)
6. Compute: ACC (overall), BWT (forgetting), FWT (improvement)
7. Compare: does SLAO+MVA beat naive on both axes?

THE 4 TASKS (from TRACE, verified to work on 350M)
--------------------------------------------------
1. C-STANCE  — stance classification (A/B/C)
2. FOMC      — finance classification (A/B/C)
3. NumGLUE-cm — math word problems (number)
4. NumGLUE-ds — math subtraction (number)

METRICS (standard, from GEM NeurIPS 2017)
-----------------------------------------
R[i,j] = score on task j after training task i

ACC = mean of last row of R = overall performance after all training
BWT = mean(R[T,j] - R[j,j]) for j<T = how much old tasks degraded (forgetting)
     negative = forgetting, zero = no forgetting, positive = improvement
FWT = mean(R[i-1,i] - baseline[i]) for i>0 = how much new tasks improved (self-improvement)

USAGE: !python v18_trace_benchmark.py
Runtime: ~3-4 hours on T4
"""

import subprocess, sys, os, json, time, random, math, gc, re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIG
# ============================================================================

MODEL_ID = "LiquidAI/LFM2.5-350M"
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("/home/z/my-project/download")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TRACE tasks (4 that work on 350M — classification + short numeric)
TRACE_TASKS = ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"]
TRACE_VARIANT = "LLM-CL-Benchmark_500"  # 500 train per task
TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"

# LoRA
LORA_RANK = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["in_proj", "out_proj"]

# Training
TRAIN_LR = 2e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 3
MVA_EPOCHS = 3
BATCH_SIZE = 8
CONTEXT_LENGTH = 512

# MVA config
ADAPTIVE_THRESHOLD = True
CERTAINTY_PERCENTILE = 50
GEN_MAX_NEW_TOKENS = 60
GEN_TEMPERATURE = 0.7
BENCH_MAX_NEW_TOKENS = 20
BENCH_TEMPERATURE = 0.0

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
    for pkg in ["peft", "datasets", "accelerate", "scipy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    # gdown for Google Drive download
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
# TRACE DATA DOWNLOAD
# ============================================================================

def download_trace_data():
    """Download TRACE benchmark data from Google Drive."""
    trace_dir = OUTPUT_DIR / "trace_data"
    if trace_dir.exists() and any(trace_dir.iterdir()):
        print(f"  TRACE data already exists: {trace_dir}")
        return trace_dir

    print(f"  Downloading TRACE data from Google Drive...")
    import gdown
    zip_path = OUTPUT_DIR / "trace_benchmark.zip"
    gdown.download(id=TRACE_GDRIVE_ID, output=str(zip_path), quiet=False)

    print(f"  Extracting...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(OUTPUT_DIR)

    # Find the variant directory
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and TRACE_VARIANT in d.name:
            trace_dir = d
            break
    else:
        # Search nested
        for d in OUTPUT_DIR.rglob(f"*{TRACE_VARIANT}*"):
            if d.is_dir():
                trace_dir = d
                break

    print(f"  TRACE data at: {trace_dir}")
    print(f"  Available tasks: {[d.name for d in trace_dir.iterdir() if d.is_dir()]}")
    return trace_dir

def load_trace_task(trace_dir, task_name):
    """Load a TRACE task's train/test data. Returns (train_pairs, test_pairs).
    Each pair is (prompt, answer).
    """
    task_dir = trace_dir / task_name
    train_path = task_dir / "train.json"
    test_path = task_dir / "test.json"

    with open(train_path) as f:
        train_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    train_pairs = [(ex["prompt"], ex["answer"]) for ex in train_data]
    test_pairs = [(ex["prompt"], ex["answer"]) for ex in test_data]

    print(f"    {task_name}: {len(train_pairs)} train, {len(test_pairs)} test")
    if train_pairs:
        print(f"      Sample prompt: {train_pairs[0][0][:100]}...")
        print(f"      Sample answer: {train_pairs[0][1][:80]}")
    return train_pairs, test_pairs

# ============================================================================
# MODEL
# ============================================================================

def load_base():
    print(f"  Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="eager",
    )
    return model, tokenizer

def create_model():
    model, tokenizer = load_base()
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    conv_c, attn_c = 0, 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer): continue
        if any(f"layers.{idx}." in name for idx in CONV_LAYER_IDS): conv_c += 1
        elif any(f"layers.{idx}." in name for idx in ATTN_LAYER_IDS): attn_c += 1
    print(f"  LoRA: {conv_c} conv + {attn_c} attn = {conv_c+attn_c} modules (rank={LORA_RANK})")
    return model, tokenizer

# ============================================================================
# SLAO CORE
# ============================================================================

def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

def slao_extract_ortho_A(model):
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

def slao_init(model, ortho_A, prev_ft_B):
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

def slao_merge(merged_state, ft_state, task_num):
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
    """Build token stream from (prompt, answer) pairs."""
    all_tokens = []
    for prompt, answer in pairs:
        text = prompt + " " + answer + tokenizer.eos_token
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

def train_on_pairs(model, tokenizer, pairs, epochs=TASK_EPOCHS):
    """Train LoRA on (prompt, answer) pairs."""
    token_ids = build_training_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    Training stream: {len(token_ids):,} tokens, {len(dataset)} chunks")

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
            if gs % 50 == 0: print(f"      step {gs} | avg_loss={tl/gs:.4f}")
    return gs, tl

# ============================================================================
# MVA (certainty-validated self-training)
# ============================================================================

def generate(model, tokenizer, prompt, max_new_tokens=GEN_MAX_NEW_TOKENS,
             temperature=GEN_TEMPERATURE, do_sample=True):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=do_sample if temperature > 0 else False, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()

def compute_certainty(model, tokenizer, prompt, answer):
    """INTUITOR self-certainty: KL(U || p_theta) averaged over answer tokens."""
    try:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=1024, add_special_tokens=True)
        answer_ids = tokenizer(" " + answer, return_tensors="pt",
                               add_special_tokens=False)["input_ids"][0]
        full_ids = torch.cat([prompt_ids["input_ids"][0], answer_ids], dim=0).unsqueeze(0)
        inputs = {"input_ids": full_ids.to(DEVICE),
                  "attention_mask": torch.ones_like(full_ids).to(DEVICE)}
        answer_start = prompt_ids["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model(**inputs)
        answer_logits = outputs.logits[0, answer_start - 1:-1, :]
        if answer_logits.shape[0] == 0: return 0.0
        log_probs = F.log_softmax(answer_logits, dim=-1)
        vocab_size = log_probs.shape[-1]
        return (-math.log(vocab_size) - log_probs.mean(dim=-1)).mean().item()
    except:
        return 0.0

def run_mva_cycle(model, tokenizer, train_pairs, merged_state, task_num):
    """MVA cycle: generate self-validated training data from task prompts.
    Uses the task's OWN training prompts, generates answers, validates by certainty.
    """
    print(f"\n  MVA cycle: generating self-validated training data...")
    # Use task prompts, generate our own answers, validate by certainty
    all_results = []
    t_start = time.time()
    for i, (prompt, gold_answer) in enumerate(train_pairs):
        answer = generate(model, tokenizer, prompt, max_new_tokens=40,
                         temperature=0.7, do_sample=True)
        certainty = compute_certainty(model, tokenizer, prompt, answer)
        all_results.append((prompt, answer, certainty))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"    [{i+1}/{len(train_pairs)}] ~{elapsed*(len(train_pairs)-i-1)/(i+1):.0f}s remaining")

    # Adaptive threshold
    certainties = np.array([c for _, _, c in all_results])
    threshold = float(np.percentile(certainties, CERTAINTY_PERCENTILE))
    n_validated = sum(1 for c in certainties if c >= threshold)
    print(f"  Certainty: mean={certainties.mean():.2f}, threshold={threshold:.2f}")
    print(f"  Validated: {n_validated}/{len(all_results)}")

    validated_pairs = [(prompt, answer) for prompt, answer, c in all_results if c >= threshold]
    if not validated_pairs:
        print("  [SKIP] No validated pairs")
        return merged_state

    # SLAO init + train + merge
    print(f"  SLAO init + training on {len(validated_pairs)} pairs...")
    ortho_A = slao_extract_ortho_A(model)
    prev_ft_B = {k: v for k, v in merged_state.items() if "lora_B" in k} if merged_state else {}
    if merged_state:
        slao_init(model, ortho_A, prev_ft_B)

    train_on_pairs(model, tokenizer, validated_pairs, epochs=MVA_EPOCHS)

    ft_state = get_lora_state(model)
    if merged_state is None:
        merged_state = ft_state.copy()
    else:
        merged_state = slao_merge(merged_state, ft_state, task_num)
    set_lora_state(model, merged_state)

    return merged_state

# ============================================================================
# EVALUATION
# ============================================================================

def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s.-]", " ", s)
    return " ".join(s.split())

def score_answer(response, gold):
    """Score: exact match after normalization. For MC (A/B/C), extract letter."""
    response = response.strip()
    gold = gold.strip()

    # MC: extract letter
    if gold in ["A", "B", "C", "D", "E"]:
        response_upper = response.upper()[:5]
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in response_upper:
                return 1.0 if letter == gold else 0.0
        return 0.0

    # Numeric: extract last number
    if re.match(r'^[\d.-]+', gold):
        numbers = re.findall(r'[\d.]+', response)
        if numbers:
            return 1.0 if numbers[-1] == gold else 0.0
        return 0.0

    # General: normalized exact match
    return 1.0 if normalize_answer(response) == normalize_answer(gold) else 0.0

def evaluate_task(model, tokenizer, test_pairs, task_name, max_questions=200):
    """Evaluate model on a task's test set. Returns accuracy."""
    print(f"    Evaluating {task_name} ({min(len(test_pairs), max_questions)} questions)...")
    correct = 0
    total = min(len(test_pairs), max_questions)
    t_start = time.time()

    for i in range(total):
        prompt, gold = test_pairs[i]
        response = generate(model, tokenizer, prompt,
                           max_new_tokens=BENCH_MAX_NEW_TOKENS,
                           temperature=0.0, do_sample=False)
        if score_answer(response, gold):
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"      [{i+1}/{total}] acc={correct/(i+1):.3f}")

    accuracy = correct / total
    elapsed = time.time() - t_start
    print(f"    {task_name}: {correct}/{total} = {accuracy:.3f} ({elapsed:.0f}s)")
    return accuracy

def evaluate_all(model, tokenizer, test_data, task_order, trained_so_far):
    """Evaluate on all tasks seen so far. Returns dict {task: accuracy}."""
    results = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far:
            break
        results[task] = evaluate_task(model, tokenizer, test_data[task], task)
    return results

# ============================================================================
# METRICS (from GEM, NeurIPS 2017)
# ============================================================================

def compute_metrics(R_matrix, task_order, baseline_scores):
    """
    R_matrix[i][j] = score on task j after training task i (0-indexed).
    baseline_scores[j] = score on task j before any training.

    Returns: ACC, BWT, FWT, FF
    """
    T = len(task_order)

    # ACC: average accuracy after all training (last row of R)
    ACC = np.mean([R_matrix[T-1][j] for j in range(T)])

    # BWT: backward transfer = how much old tasks changed from when first learned
    # BWT = mean(R[T-1][j] - R[j][j]) for j in 0..T-2
    bwt_values = [R_matrix[T-1][j] - R_matrix[j][j] for j in range(T-1)]
    BWT = np.mean(bwt_values) if bwt_values else 0.0

    # FWT: forward transfer = how much new tasks improved over baseline
    # FWT = mean(R[i-1][i] - baseline[i]) for i in 1..T-1
    fwt_values = [R_matrix[i-1][i] - baseline_scores[task_order[i]]
                  for i in range(1, T)]
    FWT = np.mean(fwt_values) if fwt_values else 0.0

    # FF: forgetting = peak ever minus final
    # FF = mean(max_l(R[l][j]) - R[T-1][j]) for j in 0..T-2
    ff_values = [max(R_matrix[l][j] for l in range(T)) - R_matrix[T-1][j]
                 for j in range(T-1)]
    FF = np.mean(ff_values) if ff_values else 0.0

    return {
        "ACC": float(ACC),
        "BWT": float(BWT),
        "FWT": float(FWT),
        "FF": float(FF),
    }

# ============================================================================
# RUN ONE METHOD (naive or SLAO+MVA)
# ============================================================================

def run_method(method_name, model, tokenizer, train_data, test_data, task_order):
    """
    Run a continual learning method on the task sequence.
    Returns: R_matrix, baseline_scores, metrics.
    """
    print(f"\n{'#'*70}")
    print(f"# METHOD: {method_name}")
    print(f"{'#'*70}")

    T = len(task_order)

    # Baseline: evaluate before any training
    print(f"\n  Baseline evaluation (before any training)...")
    baseline_scores = {}
    for task in task_order:
        baseline_scores[task] = evaluate_task(model, tokenizer, test_data[task], task)

    # R matrix: R[i][j] = score on task j after training task i
    R = [[0.0] * T for _ in range(T)]

    # After task 0 (first task): fill R[0][0]
    # After task i: fill R[i][j] for all j <= i

    merged_state = None  # for SLAO+MVA

    for i, task in enumerate(task_order):
        print(f"\n{'='*60}")
        print(f"  Training task {i+1}/{T}: {task}")
        print(f"{'='*60}")

        train_pairs = train_data[task]

        if method_name == "naive":
            # Plain sequential SFT — no protection
            train_on_pairs(model, tokenizer, train_pairs, epochs=TASK_EPOCHS)

        elif method_name == "slao_mva":
            # SLAO init (if not first task)
            if i > 0 and merged_state:
                ortho_A = slao_extract_ortho_A(model)
                prev_ft_B = {k: v for k, v in merged_state.items() if "lora_B" in k}
                slao_init(model, ortho_A, prev_ft_B)

            # Train on task data
            train_on_pairs(model, tokenizer, train_pairs, epochs=TASK_EPOCHS)

            # MVA cycle: self-validated reinforcement
            ft_state = get_lora_state(model)
            if merged_state is None:
                merged_state = ft_state.copy()
            else:
                merged_state = slao_merge(merged_state, ft_state, i + 1)
            set_lora_state(model, merged_state)

            # MVA: generate + validate + train + merge
            merged_state = run_mva_cycle(model, tokenizer, train_pairs, merged_state, i + 2)

        # Evaluate on ALL tasks seen so far
        print(f"\n  Evaluating all tasks after training {task}...")
        for j in range(i + 1):
            score = evaluate_task(model, tokenizer, test_data[task_order[j]], task_order[j])
            R[i][j] = score

        if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # Compute metrics
    metrics = compute_metrics(R, task_order, baseline_scores)

    return R, baseline_scores, metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("V18: THE REAL TEST — TRACE Benchmark with SLAO+MVA")
    print(f"Seed: {SEED} | Tasks: {TRACE_TASKS} | Variant: {TRACE_VARIANT}")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"LoRA: rank={LORA_RANK}, targets={LORA_TARGETS}")
    print(f"Device: {DEVICE}")
    print()

    # --- Download TRACE data ---
    print("Downloading TRACE benchmark data...")
    trace_dir = download_trace_data()

    # --- Load task data ---
    print("\nLoading task data...")
    train_data = {}
    test_data = {}
    for task in TRACE_TASKS:
        train_data[task], test_data[task] = load_trace_task(trace_dir, task)

    # --- Method 1: NAIVE (sequential SFT, no protection) ---
    print(f"\n{'='*70}")
    print("METHOD 1: NAIVE (sequential SFT, no protection)")
    print(f"{'='*70}")
    model, tokenizer = create_model()
    naive_R, naive_baseline, naive_metrics = run_method(
        "naive", model, tokenizer, train_data, test_data, TRACE_TASKS)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- Method 2: SLAO+MVA ---
    print(f"\n{'='*70}")
    print("METHOD 2: SLAO+MVA (the living model mechanism)")
    print(f"{'='*70}")
    model, tokenizer = create_model()
    slao_R, slao_baseline, slao_metrics = run_method(
        "slao_mva", model, tokenizer, train_data, test_data, TRACE_TASKS)

    # --- COMPARISON ---
    print(f"\n{'='*70}")
    print("THE VERDICT: Does SLAO+MVA beat naive?")
    print(f"{'='*70}")

    print(f"\n{'Metric':<12} {'Naive':<12} {'SLAO+MVA':<12} {'Delta':<10} {'Meaning'}")
    print("-" * 70)

    for metric in ["ACC", "BWT", "FWT", "FF"]:
        n = naive_metrics[metric]
        s = slao_metrics[metric]
        d = s - n
        if metric == "ACC":
            meaning = "higher = better overall"
        elif metric == "BWT":
            meaning = "higher = less forgetting"
        elif metric == "FWT":
            meaning = "higher = more improvement"
        elif metric == "FF":
            meaning = "lower = less forgetting"
        print(f"{metric:<12} {n:<12.3f} {s:<12.3f} {d:<+10.3f} {meaning}")

    print(f"\n{'='*70}")
    # Verdict
    acc_better = slao_metrics["ACC"] > naive_metrics["ACC"]
    bwt_better = slao_metrics["BWT"] > naive_metrics["BWT"]  # less negative = better
    fwt_better = slao_metrics["FWT"] > naive_metrics["FWT"]
    ff_better = slao_metrics["FF"] < naive_metrics["FF"]  # lower = better

    wins = sum([acc_better, bwt_better, fwt_better, ff_better])
    print(f"Wins: {wins}/4 metrics")

    if wins >= 3:
        print("VERDICT: YES — SLAO+MVA is clearly better than naive.")
        print("The living model mechanism works on standard benchmarks.")
    elif wins >= 2:
        print("VERDICT: PARTIAL — SLAO+MVA helps on some metrics but not all.")
    else:
        print("VERDICT: NO — SLAO+MVA does not beat naive on this benchmark.")

    # --- R matrices for analysis ---
    print(f"\n{'='*70}")
    print("R MATRICES (score on task j after training task i)")
    print(f"{'='*70}")

    for method_name, R in [("NAIVE", naive_R), ("SLAO+MVA", slao_R)]:
        print(f"\n  {method_name}:")
        header = "After\\Test  " + "  ".join(f"{t[:8]:<10}" for t in TRACE_TASKS)
        print(f"  {header}")
        for i in range(len(TRACE_TASKS)):
            row = f"  {TRACE_TASKS[i][:8]:<10} " + "  ".join(f"{R[i][j]:<10.3f}" for j in range(len(TRACE_TASKS)))
            print(row)

    # --- Save ---
    results = {
        "seed": SEED,
        "tasks": TRACE_TASKS,
        "variant": TRACE_VARIANT,
        "naive": {"R": naive_R, "baseline": naive_baseline, "metrics": naive_metrics},
        "slao_mva": {"R": slao_R, "baseline": slao_baseline, "metrics": slao_metrics},
    }
    with open(OUTPUT_DIR / "v18_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {OUTPUT_DIR}/v18_results.json")

if __name__ == "__main__":
    main()
