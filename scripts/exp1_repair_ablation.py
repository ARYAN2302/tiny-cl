"""
Experiment 1: Repair Method Ablation — "Is PPL-gating the key innovation?"

5 conditions on the same math stream (GSM8K→MATH→AQuA→SVAMP):
  A: Naive (no repair)
  B: AVR (PPL-gated, linear-interp repair)
  C: Ungated merge (repair every task, no PPL check)
  D: TIES-merge repair (PPL-gated, TIES sign-election)
  E: Task Arithmetic repair (PPL-gated, task vector subtraction)

Datasets are downloaded as parquet directly from hf-mirror.com using urllib,
bypassing HuggingFace's broken xet CDN entirely. No `datasets` library, no
`huggingface_hub` for data loading. The `datasets` library's fsspec resolver
was the leak — it ignored HF_ENDPOINT and hit huggingface.co directly.
"""
# ============================================================================
# BOOTSTRAP — must happen before any HF/transformers import
# ============================================================================
import os, sys, subprocess, urllib.request, json, shutil
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
# Route model download (Qwen3) through hf-mirror too — main HF is xet-broken
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Pin huggingface_hub to 0.24.7 (pre-xet, has is_offline_mode for transformers<=4.44)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "huggingface_hub==0.24.7",
    "peft>=0.13.0", "accelerate>=1.0.0",
    "sentencepiece", "protobuf", "packaging",
    "pyarrow"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "git+https://github.com/ARYAN2302/tiny-cl.git"], check=True)

# Numpy ABI patch: transformers thinks torch>=2.6 needs numpy 2.x
import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False

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
MIRROR = "https://hf-mirror.com"

# ============================================================================
# Direct parquet download — no datasets lib, no fsspec, no xet
# ============================================================================
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

def _read_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(str(path)).to_pylist()

def _hf_parquet_url(repo, subdir, split):
    """Build a hf-mirror.com parquet URL for a dataset."""
    return f"{MIRROR}/datasets/{repo}/resolve/main/{subdir}/{split}-00000-of-00001.parquet"

def load_gsm8k(n, split="train"):
    fn = "test" if split == "test" else "train"
    url = _hf_parquet_url("openai/gsm8k", "main", fn)
    cache = DATA_CACHE / f"gsm8k_{fn}.parquet"
    rows = _read_parquet(_download(url, cache))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_math(n, split="train"):
    fn = "test" if split == "test" else "train"
    url = _hf_parquet_url("EleutherAI/hendrycks_math", "algebra", fn)
    cache = DATA_CACHE / f"math_algebra_{fn}.parquet"
    rows = _read_parquet(_download(url, cache))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        q = ex["problem"]; sol = ex["solution"]
        m = re.findall(r'\\boxed\{([^}]+)\}', sol)
        gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((f"Solve. End with \\boxed{{answer}}.\n\n{q}", sol, gold))
    return pairs

def load_aqua(n, split="train"):
    fn = "validation" if split == "test" else "train"
    url = _hf_parquet_url("deepmind/aqua_rat", "raw", fn)
    cache = DATA_CACHE / f"aqua_{fn}.parquet"
    rows = _read_parquet(_download(url, cache))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    letters = ["A", "B", "C", "D", "E"]
    for ex in rows[:n]:
        q = ex["question"]; opts = ex["options"]; correct = ex["correct"]
        rationale = ex.get("rationale", "")
        cleaned = []
        for i, o in enumerate(opts):
            o = str(o).strip()
            if len(o) >= 2 and o[0].upper() == letters[i] and o[1] in ").:":
                o = o[2:].strip()
            cleaned.append(o)
        opt_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, cleaned))
        pairs.append((f"{q}\n{opt_text}\n\nAnswer with letter:",
                      f"{rationale}\nAnswer: {correct}", correct))
    return pairs

def load_svamp(n, split="train"):
    fn = "test" if split == "test" else "train"
    url = _hf_parquet_url("ChilleD/SVAMP", "data", fn)
    cache = DATA_CACHE / f"svamp_{fn}.parquet"
    rows = _read_parquet(_download(url, cache))
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
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

print("\nLoading data (500 ex/task) from hf-mirror.com parquet...", flush=True)
gsm8k_tr = load_gsm8k(500); gsm8k_te = load_gsm8k(100, "test")
math_tr  = load_math(500);  math_te  = load_math(100, "test")
aqua_tr  = load_aqua(500);  aqua_te  = load_aqua(100, "test")
svamp_tr = load_svamp(500); svamp_te = load_svamp(100, "test")
print(f"  loaded: gsm8k={len(gsm8k_tr)}/{len(gsm8k_te)} math={len(math_tr)}/{len(math_te)} "
      f"aqua={len(aqua_tr)}/{len(aqua_te)} svamp={len(svamp_tr)}/{len(svamp_te)}", flush=True)

tasks_data = [
    ("gsm8k", gsm8k_tr, gsm8k_te),
    ("math",  math_tr,  math_te),
    ("aqua",  aqua_tr,  aqua_te),
    ("svamp", svamp_tr, svamp_te),
]

COMMON = dict(
    model="Qwen/Qwen3-1.7B",
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
