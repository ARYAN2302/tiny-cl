"""
Experiment 0: Two-Stream + AVR on Qwen3-1.7B math stream.

Tests whether the two-stream variant (hippocampus/neocortex + KL distill)
beats AVR alone on the headline model + stream.

Uses the avr package: avr.run(two_stream=True)
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
import json, re, random
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

def load_gsm8k(n=5000):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_gsm8k_eval(n=200):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_math(n=5000):
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
        m = re.findall(r'\\boxed\{([^}]+)\}', sol)
        gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((f"Solve. End with \\boxed{{answer}}.\n\n{q}", sol, gold))
    return pairs

def load_math_eval(n=200):
    from datasets import load_dataset
    try:
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    except:
        ds = load_dataset("lighteval/MATH", "algebra", split="test")
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

def load_aqua(n=5000):
    from datasets import load_dataset
    ds = load_dataset("deepmind/aqua_rat", "raw", split="train")
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

def load_aqua_eval(n=200):
    from datasets import load_dataset
    ds = load_dataset("deepmind/aqua_rat", "raw", split="validation")
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

def load_svamp(n=5000):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="train")
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

def load_svamp_eval(n=200):
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="test")
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

# Math answer scorer
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
print("EXP 0: Two-Stream + AVR on Qwen3-1.7B math stream", flush=True)
print("="*70, flush=True)

print("\nLoading data...", flush=True)
gsm8k_tr = load_gsm8k(5000); gsm8k_te = load_gsm8k_eval(200)
math_tr = load_math(5000); math_te = load_math_eval(200)
aqua_tr = load_aqua(5000); aqua_te = load_aqua_eval(200)
svamp_tr = load_svamp(5000); svamp_te = load_svamp_eval(200)
print(f"  gsm8k: {len(gsm8k_tr)} train, {len(gsm8k_te)} eval", flush=True)
print(f"  math:  {len(math_tr)} train, {len(math_te)} eval", flush=True)
print(f"  aqua:  {len(aqua_tr)} train, {len(aqua_te)} eval", flush=True)
print(f"  svamp: {len(svamp_tr)} train, {len(svamp_te)} eval", flush=True)

result = avr.run(
    model="Qwen/Qwen3-1.7B",
    tasks=[
        ("gsm8k", gsm8k_tr, gsm8k_te),
        ("math_algebra", math_tr, math_te),
        ("aqua_rat", aqua_tr, aqua_te),
        ("svamp", svamp_tr, svamp_te),
    ],
    lora_rank=128,
    lora_alpha=128,
    lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
    epochs=3,
    lr=2e-4,
    batch_size=4,
    grad_accum=4,
    ctx_len=512,
    drift_threshold=1.15,
    repair_alpha=0.1,
    max_repair_steps=10,
    two_stream=True,
    scorer=math_scorer,
    seed=42,
)

print(f"\n{'='*70}", flush=True)
print(f"RESULTS: Two-Stream + AVR", flush=True)
print(f"  ACC: {result['acc']:.3f}", flush=True)
print(f"  BWT: {result['bwt']:+.3f}", flush=True)
print(f"  FF:  {result['ff']:.3f}", flush=True)
print(f"  Repairs: {result['repairs']}", flush=True)
print(f"{'='*70}", flush=True)

with open(OUTPUT_DIR / "exp0_twostream_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp0_twostream_results.json", flush=True)
