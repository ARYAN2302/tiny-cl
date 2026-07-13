"""
Experiment 4: Cross-domain on Qwen3-1.7B — Code → Math → Instruct → Science.

Tests whether AVR works when domains are maximally unrelated.
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
import json, re, random
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

def load_code(n=500):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        prompt = ex.get("instruction", "")
        completion = ex.get("output", "")
        gold = completion[:50]
        pairs.append((f"Write Python code for:\n{prompt}\n\nCode:", completion, gold))
    return pairs

def load_gsm8k(n=500):
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

def load_gsm8k_eval(n=100):
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

def load_alpaca(n=500):
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        instruction = ex["instruction"]
        inp = ex.get("input", "")
        output = ex["output"]
        prompt = f"{instruction}\n{inp}\n\nAnswer:" if inp else f"{instruction}\n\nAnswer:"
        pairs.append((prompt, output, output[:50]))
    return pairs

def load_sciq(n=500):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]
        correct = ex["correct_answer"]
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        sr = random.Random(hash(q) & 0xffffffff)
        sr.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        gold = letters[opts.index(correct)]
        prompt = f"{q}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n\nAnswer:"
        pairs.append((prompt, gold, gold))
    return pairs

def load_sciq_eval(n=100):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="test")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]
        correct = ex["correct_answer"]
        opts = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        sr = random.Random(hash(q) & 0xffffffff)
        sr.shuffle(opts)
        letters = ["A", "B", "C", "D"]
        gold = letters[opts.index(correct)]
        prompt = f"{q}\nA. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\n\nAnswer:"
        pairs.append((prompt, gold, gold))
    return pairs

# Cross-domain scorer: handles both MCQ and text
def cross_scorer(response, gold):
    resp = response.strip()
    g = gold.strip()
    if g in ["A","B","C","D","E"]:
        tail = resp[-100:].upper()
        m = re.search(r'\b([A-E])\b', tail)
        if m: return 1.0 if m.group(1) == g else 0.0
        if resp and resp[0].upper() in "ABCDE": return 1.0 if resp[0].upper() == g else 0.0
        return 0.0
    resp_n = resp.lower().strip()
    g_n = g.lower().strip()
    if resp_n == g_n: return 1.0
    if g_n in resp_n or resp_n in g_n: return 1.0
    return 0.0

print("="*70, flush=True)
print("EXP 4: Cross-domain (Code → Math → Instruct → Science)", flush=True)
print("="*70, flush=True)

print("\nLoading data...", flush=True)
code_tr = load_code(500)
gsm8k_tr = load_gsm8k(500); gsm8k_te = load_gsm8k_eval(100)
alpaca_tr = load_alpaca(500)
sciq_tr = load_sciq(500); sciq_te = load_sciq_eval(100)

# Eval sets: use last 50 of train for code/alpaca (no separate test)
code_te = code_tr[-50:]; code_tr = code_tr[:-50]
alpaca_te = alpaca_tr[-50:]; alpaca_tr = alpaca_tr[:-50]

print(f"  code:    {len(code_tr)} train, {len(code_te)} eval", flush=True)
print(f"  gsm8k:   {len(gsm8k_tr)} train, {len(gsm8k_te)} eval", flush=True)
print(f"  alpaca:  {len(alpaca_tr)} train, {len(alpaca_te)} eval", flush=True)
print(f"  sciq:    {len(sciq_tr)} train, {len(sciq_te)} eval", flush=True)

# Run AVR
result = avr.run(
    model="Qwen/Qwen3-1.7B",
    tasks=[
        ("code", code_tr, code_te),
        ("gsm8k", gsm8k_tr, gsm8k_te),
        ("alpaca", alpaca_tr, alpaca_te),
        ("sciq", sciq_tr, sciq_te),
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
    two_stream=False,
    scorer=cross_scorer,
    seed=42,
)

print(f"\n{'='*70}", flush=True)
print(f"RESULTS: Cross-domain AVR", flush=True)
print(f"  ACC: {result['acc']:.3f}", flush=True)
print(f"  BWT: {result['bwt']:+.3f}", flush=True)
print(f"  FF:  {result['ff']:.3f}", flush=True)
print(f"  Repairs: {result['repairs']}", flush=True)
print(f"{'='*70}", flush=True)

with open(OUTPUT_DIR / "exp4_crossdomain_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp4_crossdomain_results.json", flush=True)
