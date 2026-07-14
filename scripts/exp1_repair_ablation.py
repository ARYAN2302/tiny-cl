"""
Experiment 1: Repair Method Ablation — "Is PPL-gating the key innovation?"

5 conditions on the same math stream (GSM8K→MATH→AQuA→SVAMP):
  A: Naive (no repair)
  B: AVR (PPL-gated, linear-interp repair)
  C: Ungated merge (repair every task, no PPL check)
  D: TIES-merge repair (PPL-gated, TIES sign-election)
  E: Task Arithmetic repair (PPL-gated, task vector subtraction)

Datasets are downloaded as JSON/JSONL directly from GitHub raw URLs via
urllib, bypassing HuggingFace entirely. The model (Qwen3-1.7B) is
downloaded from ModelScope (Alibaba's model hub — they make Qwen, so it's
the canonical source, and it doesn't use xet CDN). No HuggingFace
infrastructure is involved in any download.
"""
# ============================================================================
# BOOTSTRAP — must happen before any HF/transformers import
# ============================================================================
import os, sys, subprocess, urllib.request, json, shutil
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Route ALL HF traffic (model download) through hf-mirror.com.
# Newer huggingface_hub (>=0.28) respects this properly — no xet, no 403.
# xet only kicks in on huggingface.co itself; mirrors don't use it.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Install deps — do NOT pin huggingface_hub. Let pip resolve it to match
# the installed transformers (5.0.0 needs hub>=1.3.0). Hub 0.24.7 was too
# old and didn't respect HF_ENDPOINT for metadata checks.
# Install modelscope for model download — bypasses HF xet CDN entirely.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "accelerate>=1.0.0",
    "sentencepiece", "protobuf", "packaging",
    "modelscope"], check=True)
# torchao breaks numpy ABI on Kaggle — remove it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
# hf-xet hijacks download routing — remove it (belt and suspenders)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet"], check=False)
# Install/refresh avr from git
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "git+https://github.com/ARYAN2302/tiny-cl.git"], check=True)

# Numpy ABI patch — only needed for older transformers that think torch>=2.6
# requires numpy 2.x. Wrapped in try/except because transformers 5.0.0 may
# not have this attribute.
try:
    import transformers.utils.import_utils as _iu
    _iu._is_torch_greater_or_equal_than_2_6 = False
    _iu.is_torch_greater_or_equal_than_2_6 = lambda: False
except (AttributeError, ImportError):
    pass

# ============================================================================
# Real imports
# ============================================================================
import avr
import re, random, math, gc, torch, numpy as np
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE = OUTPUT_DIR / "data_cache"
DATA_CACHE.mkdir(parents=True, exist_ok=True)
SEED = 42

# ============================================================================
# DATA LOADERS — direct GitHub raw downloads. NO HuggingFace involved.
# All URLs verified HTTP 200, no xet, no auth, no redirects to broken CDN.
# ============================================================================
GSM8K_BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"
MATH_URL   = "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/main/math_full.json"
AQUA_BASE  = "https://raw.githubusercontent.com/google-deepmind/AQuA/master"
SVAMP_URL  = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"

def _download(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "avr-cl/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  -> {dest.stat().st_size} bytes", flush=True)
    return str(dest)

def load_gsm8k(n, split="train"):
    fn = "test" if split == "test" else "train"
    cache = DATA_CACHE / f"gsm8k_{fn}.jsonl"
    path = _download(f"{GSM8K_BASE}/{fn}.jsonl", cache)
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_math(n, split="train"):
    """MATH dataset — 12500 combined records, split ourselves by seed.
    The rasbt/math_full_minus_math500 mirror has no train/test split, so we
    use a deterministic shuffle: same shuffle for both 'train' and 'test'
    calls, then take the first n for train and next 100 for test."""
    cache = DATA_CACHE / "math_full.json"
    path = _download(MATH_URL, cache)
    with open(path) as f:
        rows = json.load(f)
    rng = random.Random(SEED); rng.shuffle(rows)
    if split == "test":
        rows = rows[500:600]  # next 100 after the train slice
    else:
        rows = rows[:n]
    pairs = []
    for ex in rows[:n] if split == "train" else rows:
        q = ex["problem"]; sol = ex["solution"]
        # MATH has an explicit 'answer' field — use it directly when possible
        gold = ex.get("answer", "").strip()
        if not gold:
            m = re.findall(r'\\boxed\{([^}]+)\}', sol)
            gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((f"Solve. End with \\boxed{{answer}}.\n\n{q}", sol, gold))
    return pairs

def load_aqua(n, split="train"):
    fn = "dev" if split == "test" else "train"
    cache = DATA_CACHE / f"aqua_{fn}.json"
    path = _download(f"{AQUA_BASE}/{fn}.json", cache)
    # AQuA files are JSONL (one obj per line) despite the .json extension
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    letters = ["A", "B", "C", "D", "E"]
    for ex in rows[:n]:
        q = ex["question"]; opts = ex["options"]; correct = ex["correct"]
        rationale = ex.get("rationale", "")
        cleaned = []
        for i, o in enumerate(opts):
            o = str(o).strip()
            # AQuA options are like "A)32400" — strip the letter prefix
            if len(o) >= 2 and o[0].upper() == letters[i] and o[1] in ").:":
                o = o[2:].strip()
            cleaned.append(o)
        opt_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, cleaned))
        pairs.append((f"{q}\n{opt_text}\n\nAnswer with letter:",
                      f"{rationale}\nAnswer: {correct}", correct))
    return pairs

def load_svamp(n, split="train"):
    # SVAMP is a single 1000-item JSON file — no train/test split.
    # Use deterministic shuffle, take first n for train, next 100 for test.
    cache = DATA_CACHE / "svamp.json"
    path = _download(SVAMP_URL, cache)
    with open(path) as f:
        rows = json.load(f)
    rng = random.Random(SEED); rng.shuffle(rows)
    if split == "test":
        rows = rows[500:600]
    else:
        rows = rows[:n]
    pairs = []
    for ex in rows[:n] if split == "train" else rows:
        body = ex.get("Body", ""); question = ex.get("Question", "")
        answer = ex.get("Answer", ""); equation = ex.get("Equation", "")
        full_q = f"{body} {question}".strip()
        try:
            gf = float(answer)
            gold = str(int(gf)) if gf == int(gf) else str(gf)
        except Exception:
            gold = str(answer)
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{full_q}",
                      f"Equation: {equation}\n#### {gold}", gold))
    return pairs

def normalize_math(s):
    s = s.strip().replace('$','').replace('\\','').replace('!','').replace(',','')
    s = s.replace('{','').replace('}','').replace(' ','')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except Exception:
        return s.lower()

def extract_answer(response, is_mcq=False):
    response = response.strip()
    if is_mcq:
        tail = response[-100:].upper()
        m = re.search(r'\b([A-E])\b', tail)
        if m: return m.group(1)
        if response and response[0].upper() in "ABCDE": return response[0].upper()
        return response[:1]
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

def math_scorer(response, gold):
    is_mcq = gold in ["A","B","C","D","E"]
    if is_mcq:
        return 1.0 if extract_answer(response, is_mcq=True).upper() == gold.upper() else 0.0
    return 1.0 if normalize_math(extract_answer(response)) == normalize_math(gold) else 0.0

# ============================================================================
# CUSTOM REPAIR OPERATORS — same signature as avr.repair.repair
#   fn(model, snapshot, alpha, device) -> int (params touched)
# ============================================================================
def repair_linear(model, snapshot, alpha=0.1, device="cuda"):
    """AVR default: linear interpolation toward snapshot.
       current <- (1-alpha)*current + alpha*snapshot"""
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name and name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))
            n += 1
    return n

def repair_ties(model, snapshot, alpha=0.1, device="cuda"):
    """TIES-merge style repair.
    delta = current - snapshot (what drifted)
    1. Trim: zero out bottom 20% magnitude deltas
    2. Elect: per-tensor majority sign wins
    3. Merge: current <- current - alpha * elected_delta
    """
    n = 0
    for name, p in model.named_parameters():
        if "lora_" not in name or name not in snapshot:
            continue
        snap_val = snapshot[name].to(device)
        delta = p.data - snap_val
        if delta.numel() == 0:
            continue
        threshold = torch.quantile(delta.abs().flatten().float(), 0.2).to(delta.dtype)
        delta = torch.where(delta.abs() < threshold, torch.zeros_like(delta), delta)
        sign_sum = float(torch.sign(delta).sum().item())
        elected_sign = 1.0 if sign_sum >= 0 else -1.0
        delta = torch.where(torch.sign(delta) == elected_sign, delta, torch.zeros_like(delta))
        p.data.copy_(p.data - alpha * delta)
        n += 1
    return n

def repair_task_arithmetic(model, snapshot, alpha=0.1, device="cuda"):
    """Task Arithmetic repair.
    task_vector = current - snapshot
    current <- current - alpha * task_vector
    """
    n = 0
    for name, p in model.named_parameters():
        if "lora_" not in name or name not in snapshot:
            continue
        snap_val = snapshot[name].to(device)
        task_vector = p.data - snap_val
        p.data.copy_(p.data - alpha * task_vector)
        n += 1
    return n

# ============================================================================
# MAIN — Run 5 conditions
# ============================================================================
print("="*70, flush=True)
print("EXP 1: Repair Method Ablation", flush=True)
print("A: Naive | B: AVR | C: Ungated | D: TIES | E: TaskArith", flush=True)
print("="*70, flush=True)

print("\nLoading data (500 ex/task) from GitHub raw...", flush=True)
gsm8k_tr = load_gsm8k(500); gsm8k_te = load_gsm8k(100, "test")
math_tr  = load_math(500);  math_te  = load_math(100, "test")
aqua_tr  = load_aqua(500);  aqua_te  = load_aqua(100, "test")
svamp_tr = load_svamp(500); svamp_te = load_svamp(100, "test")
print(f"  loaded: gsm8k={len(gsm8k_tr)}/{len(gsm8k_te)} math={len(math_tr)}/{len(math_te)} "
      f"aqua={len(aqua_tr)}/{len(aqua_te)} svamp={len(svamp_tr)}/{len(svamp_te)}", flush=True)

# Download model from ModelScope (Alibaba's hub — hosts Qwen, no xet, no HF)
print("\nDownloading Qwen3-1.7B from ModelScope...", flush=True)
from modelscope import snapshot_download
MODEL_PATH = snapshot_download("Qwen/Qwen3-1.7B", cache_dir=OUTPUT_DIR / "model_cache")
print(f"  Model cached at: {MODEL_PATH}", flush=True)

tasks_data = [
    ("gsm8k", gsm8k_tr, gsm8k_te),
    ("math",  math_tr,  math_te),
    ("aqua",  aqua_tr,  aqua_te),
    ("svamp", svamp_tr, svamp_te),
]

COMMON = dict(
    model=MODEL_PATH,
    tasks=tasks_data,
    lora_rank=128,
    lora_targets=["q_proj","k_proj","v_proj","o_proj"],
    epochs=3, batch_size=4, grad_accum=4, ctx_len=512,
    scorer=math_scorer, seed=SEED,
)

results = {}

def run_condition(label, desc, **kwargs):
    print(f"\n{'#'*60}\n# Condition {label}: {desc}\n{'#'*60}", flush=True)
    try:
        r = avr.run(**{**COMMON, **kwargs})
        print(f"  {label}: BWT={r['bwt']:+.3f}  ACC={r['acc']:.3f}  Repairs={r['repairs']}", flush=True)
        results[label] = r
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {label} failed: {e}", flush=True)
        results[label] = {"error": str(e)}
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# A: Naive — threshold=999 so PPL gate never trips
run_condition("A_naive", "Naive (no repair)",
    drift_threshold=999.0, repair_alpha=0.0, max_repair_steps=0)

# B: AVR — PPL-gated, linear interpolation toward snapshot
run_condition("B_avr", "AVR (PPL-gated, linear-interp)",
    drift_threshold=1.15, repair_alpha=0.1, max_repair_steps=10,
    repair_fn=repair_linear)

# C: Ungated merge — threshold=1.0 so gate fires after every task
run_condition("C_ungated", "Ungated merge (no PPL check)",
    drift_threshold=1.0, repair_alpha=0.1, max_repair_steps=10,
    repair_fn=repair_linear)

# D: TIES — PPL-gated, TIES sign-election repair
run_condition("D_ties", "TIES (PPL-gated, sign-election)",
    drift_threshold=1.15, repair_alpha=0.1, max_repair_steps=10,
    repair_fn=repair_ties)

# E: Task Arithmetic — PPL-gated, task vector subtraction
run_condition("E_taskarith", "Task Arithmetic (PPL-gated)",
    drift_threshold=1.15, repair_alpha=0.1, max_repair_steps=10,
    repair_fn=repair_task_arithmetic)

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*70}", flush=True)
print("REPAIR ABLATION RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n{'Condition':<28} {'BWT':<10} {'ACC':<10} {'Repairs':<10}", flush=True)
print("-"*58, flush=True)
for cond, label in [("A_naive","Naive"),
                    ("B_avr","AVR (PPL-gated)"),
                    ("C_ungated","Ungated merge"),
                    ("D_ties","TIES"),
                    ("E_taskarith","Task Arith")]:
    r = results.get(cond, {})
    if "bwt" in r:
        print(f"{label:<28} {r['bwt']:<+10.3f} {r['acc']:<10.3f} {r['repairs']:<10}", flush=True)
    else:
        print(f"{label:<28} {'FAIL':<10}", flush=True)
print("-"*58, flush=True)

def strip(r):
    if "error" in r: return r
    return {k: v for k, v in r.items() if k != "R"}

with open(OUTPUT_DIR / "exp1_repair_ablation.json", "w") as f:
    json.dump({k: strip(v) for k, v in results.items()}, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp1_repair_ablation.json", flush=True)
