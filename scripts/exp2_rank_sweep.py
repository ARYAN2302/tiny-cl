"""
Experiment 2: LoRA Rank Sensitivity Sweep.

Tests whether AVR adds value across PEFT configs (r=8 to r=128).
Addresses Bubeck et al. (ICLR 2025): "LoRA Learns Less and Forgets Less."

5 ranks x 2 conditions (Naive vs AVR) = 10 runs.
Uses the avr package.
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "datasets>=3.0.0", "accelerate>=1.0.0",
    "sentencepiece", "protobuf", "packaging"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
    "git+https://github.com/ARYAN2302/tiny-cl.git"], check=True)

import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False

import avr
import json, re, random, gc, torch
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
RANKS = [8, 16, 32, 64, 128]

# Data loaders (abbreviated — same math stream, 500 ex/task for speed)
def load_gsm8k(n, split="train"):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_math(n, split="train"):
    from datasets import load_dataset
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split=split)
    except:
        ds = load_dataset("lighteval/MATH", "algebra", split=split)
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["problem"]; sol = ex["solution"]
        m = re.findall(r'\\boxed\{([^}]+)\}', sol)
        gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((f"Solve. End with \\boxed{{answer}}.\n\n{q}", sol, gold))
    return pairs

def load_aqua(n, split="train"):
    from datasets import load_dataset
    split_name = "validation" if split == "test" else split
    ds = load_dataset("deepmind/aqua_rat", "raw", split=split_name)
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    letters = ["A", "B", "C", "D", "E"]
    for ex in all_ex[:n]:
        q = ex["question"]; opts = ex["options"]; correct = ex["correct"]
        rationale = ex.get("rationale", "")
        cleaned = []
        for i, o in enumerate(opts):
            o = str(o).strip()
            if len(o) >= 2 and o[0].upper() == letters[i] and o[1] in ").:":
                o = o[2:].strip()
            cleaned.append(o)
        opt_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, cleaned))
        pairs.append((f"{q}\n{opt_text}\n\nAnswer with letter:", f"{rationale}\nAnswer: {correct}", correct))
    return pairs

def load_svamp(n, split="train"):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split=split)
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        body = ex.get("Body", ""); question = ex.get("Question", "")
        answer = ex.get("Answer", ""); equation = ex.get("Equation", "")
        full_q = f"{body} {question}".strip()
        try:
            gf = float(answer)
            gold = str(int(gf)) if gf == int(gf) else str(gf)
        except: gold = str(answer)
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{full_q}",
                      f"Equation: {equation}\n#### {gold}", gold))
    return pairs

def normalize_math(s):
    s = s.strip().replace('$','').replace('\\','').replace('!','').replace(',','')
    s = s.replace('{','').replace('}','').replace(' ','')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except: return s.lower()

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
# MAIN
# ============================================================================
print("="*70, flush=True)
print("EXP 2: LoRA Rank Sensitivity Sweep", flush=True)
print(f"Ranks: {RANKS}", flush=True)
print("="*70, flush=True)

print("\nLoading data (500 ex/task)...", flush=True)
tasks_data = [
    ("gsm8k", load_gsm8k(500), load_gsm8k(100, "test")),
    ("math", load_math(500), load_math(100, "test")),
    ("aqua", load_aqua(500), load_aqua(100, "test")),
    ("svamp", load_svamp(500), load_svamp(100, "test")),
]

all_results = {}

for rank in RANKS:
    print(f"\n{'='*60}", flush=True)
    print(f"  RANK r={rank}", flush=True)
    print(f"{'='*60}", flush=True)

    # Naive
    print(f"  Naive...", flush=True)
    try:
        r_naive = avr.run(
            model="Qwen/Qwen3-1.7B", tasks=tasks_data,
            lora_rank=rank, lora_alpha=rank,
            lora_targets=["q_proj","k_proj","v_proj","o_proj"],
            epochs=3, batch_size=4, grad_accum=4, ctx_len=512,
            drift_threshold=999.0, repair_alpha=0.0, max_repair_steps=0,
            scorer=math_scorer, seed=SEED)
        print(f"  Naive r={rank}: BWT={r_naive['bwt']:+.3f}", flush=True)
    except Exception as e:
        print(f"  Naive r={rank} failed: {e}", flush=True)
        r_naive = {"error": str(e), "bwt": None}
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # AVR
    print(f"  AVR...", flush=True)
    try:
        r_avr = avr.run(
            model="Qwen/Qwen3-1.7B", tasks=tasks_data,
            lora_rank=rank, lora_alpha=rank,
            lora_targets=["q_proj","k_proj","v_proj","o_proj"],
            epochs=3, batch_size=4, grad_accum=4, ctx_len=512,
            drift_threshold=1.15, repair_alpha=0.1, max_repair_steps=10,
            scorer=math_scorer, seed=SEED)
        print(f"  AVR r={rank}: BWT={r_avr['bwt']:+.3f}", flush=True)
    except Exception as e:
        print(f"  AVR r={rank} failed: {e}", flush=True)
        r_avr = {"error": str(e), "bwt": None}
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    all_results[f"r{rank}"] = {
        "naive_bwt": r_naive.get("bwt"),
        "avr_bwt": r_avr.get("bwt"),
        "naive_acc": r_naive.get("acc"),
        "avr_acc": r_avr.get("acc"),
        "avr_repairs": r_avr.get("repairs"),
    }

# Summary table
print(f"\n{'='*70}", flush=True)
print("RANK SENSITIVITY RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n{'Rank':<8} {'Naive BWT':<12} {'AVR BWT':<12} {'ΔBWT':<12} {'Repairs':<10}", flush=True)
print("-"*54, flush=True)
for rank in RANKS:
    r = all_results.get(f"r{rank}", {})
    nb = r.get("naive_bwt"); ab = r.get("avr_bwt")
    delta = (ab - nb) if nb is not None and ab is not None else None
    repairs = r.get("avr_repairs", "-")
    print(f"r={rank:<5} {nb:<+12.3f} {ab:<+12.3f} {delta:<+12.3f} {repairs:<10}" if nb and ab
          else f"r={rank:<5} {'N/A':<12} {'N/A':<12} {'N/A':<12} {repairs:<10}", flush=True)
print("-"*54, flush=True)

with open(OUTPUT_DIR / "exp2_rank_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp2_rank_sweep.json", flush=True)
