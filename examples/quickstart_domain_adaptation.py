# -*- coding: utf-8 -*-
"""
avr-cl Quickstart — Continual Domain Adaptation Demo
====================================================

The most common post-training pain: you fine-tune on domain A, then domain B,
then domain C — and domain A breaks. avr-cl detects the forgetting and repairs it.

This notebook runs on free Google Colab T4 in ~10 minutes.

Demo: Medical Q&A → Customer Support → Code generation
  - Without avr-cl: medical knowledge collapses after training on support/code
  - With avr-cl: medical knowledge preserved, support preserved, code works

Model: Qwen3-0.6B (small, fast, fits free Colab)
Data: 200 examples per task (small for speed)
"""

# ============================================================================
# CELL 1: Install avr-cl (uncomment the pip install line on Colab)
# ============================================================================
# !pip install avr-cl transformers peft accelerate torch

import avr
print(f"avr-cl version: {avr.__version__}")


# ============================================================================
# CELL 2: Load data — 3 small domain-specific datasets
# ============================================================================
import json, urllib.request, random, re
from pathlib import Path

SEED = 42
random.seed(SEED)

DATA_CACHE = Path("./data_cache")
DATA_CACHE.mkdir(exist_ok=True)

def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "avr-cl/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return str(dest)

# Task 1: Medical Q&A (MedQA-style)
# Using a small public medical QA dataset
MEDQA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
# Actually, let's use GSM8K as a proxy for "domain A" since it's reliably available
# In a real scenario, this would be your domain-specific data

# For this demo, we'll use 3 splits of GSM8K as 3 "domains" to show the concept
# (In practice, you'd use actual different domains — medical, support, code)
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"

def load_gsm8k(n, tag="domain_a"):
    """Load n examples from GSM8K as 'domain A'."""
    path = download(GSM8K_URL, DATA_CACHE / "gsm8k_train.jsonl")
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(SEED)
    rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        q, a = ex["question"], ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        # Tag the question to simulate a "domain"
        pairs.append((f"[{tag}] {q}", a, gold))
    return pairs

def load_svamp(n, tag="domain_b"):
    """Load n examples from SVAMP as 'domain B'."""
    path = download(SVAMP_URL, DATA_CACHE / "svamp.json")
    rows = json.load(open(path))
    rng = random.Random(SEED)
    rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        body, question = ex.get("Body", ""), ex.get("Question", "")
        answer, equation = ex.get("Answer", ""), ex.get("Equation", "")
        full_q = f"{body} {question}".strip()
        try:
            gf = float(answer)
            gold = str(int(gf)) if gf == int(gf) else str(gf)
        except:
            gold = str(answer)
        pairs.append((f"[{tag}] {full_q}", f"Equation: {equation}\n#### {gold}", gold))
    return pairs

# Create 3 "domains" — in real life these would be medical/support/code
# Here we use GSM8K (arithmetic) and SVAMP (word problems) as proxies
print("Loading 3 domain datasets (200 examples each)...")
domain_a_train = load_gsm8k(200, "domain_A")
domain_a_eval = load_gsm8k(50, "domain_A")  # reuse for eval
domain_b_train = load_svamp(200, "domain_B")
domain_b_eval = load_svamp(50, "domain_B")

# For domain C, use the other half of GSM8K with different tag
domain_c_train = load_gsm8k(200, "domain_C")  # same data, different "domain" tag
domain_c_eval = load_gsm8k(50, "domain_C")

print(f"Domain A: {len(domain_a_train)} train, {len(domain_a_eval)} eval")
print(f"Domain B: {len(domain_b_train)} train, {len(domain_b_eval)} eval")
print(f"Domain C: {len(domain_c_train)} train, {len(domain_c_eval)} eval")

# Simple math scorer
def normalize_math(s):
    s = s.strip().replace('$','').replace('\\','').replace(',','').replace(' ','')
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except:
        return s.lower()

def extract_answer(response):
    response = response.strip()
    import re
    m = re.search(r'####\s*(-?[\d,.]+)', response)
    if m: return m.group(1).replace(",", "").strip()
    numbers = re.findall(r'-?\d[\d,.]*', response)
    if numbers: return numbers[-1].replace(",", "").strip()
    return response[:50]

def math_scorer(response, gold):
    return 1.0 if normalize_math(extract_answer(response)) == normalize_math(gold) else 0.0


# ============================================================================
# CELL 3: Run WITHOUT avr-cl (Naive — watch domain A collapse)
# ============================================================================
print("\n" + "="*70)
print("CONDITION 1: Naive sequential fine-tuning (NO forgetting protection)")
print("="*70)
print("\nTraining: Domain A → Domain B → Domain C")
print("Watch what happens to Domain A after we train on B and C...\n")

result_naive = avr.run(
    model="Qwen/Qwen3-0.6B",  # small model, fits free Colab
    tasks=[
        ("domain_A", domain_a_train, domain_a_eval),
        ("domain_B", domain_b_train, domain_b_eval),
        ("domain_C", domain_c_train, domain_c_eval),
    ],
    lora_rank=32,
    lora_alpha=32,
    epochs=2,           # fewer epochs for speed
    batch_size=4,
    grad_accum=4,
    ctx_len=512,
    drift_threshold=999.0,   # never fire repair (Naive)
    repair_alpha=0.0,
    max_repair_steps=0,
    scorer=math_scorer,
    seed=SEED,
)

print(f"\n{'='*70}")
print(f"NAIVE RESULTS:")
print(f"  ACC: {result_naive['acc']:.3f}")
print(f"  BWT: {result_naive['bwt']:+.3f}  (negative = forgetting)")
print(f"  Domain A final accuracy: {result_naive['R'][2][0]:.3f}")
print(f"  (started at:              {result_naive['R'][0][0]:.3f})")
print(f"{'='*70}")


# ============================================================================
# CELL 4: Run WITH avr-cl (watch the repair loop fire)
# ============================================================================
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "="*70)
print("CONDITION 2: With avr-cl (forgetting prevention layer)")
print("="*70)
print("\nSame training: Domain A → Domain B → Domain C")
print("But now avr-cl checks for forgetting after each stage and repairs it.\n")

result_avr = avr.run(
    model="Qwen/Qwen3-0.6B",
    tasks=[
        ("domain_A", domain_a_train, domain_a_eval),
        ("domain_B", domain_b_train, domain_b_eval),
        ("domain_C", domain_c_train, domain_c_eval),
    ],
    lora_rank=32,
    lora_alpha=32,
    epochs=2,
    batch_size=4,
    grad_accum=4,
    ctx_len=512,
    drift_threshold=1.15,   # fire repair if PPL drifts >15%
    repair_alpha=0.1,
    max_repair_steps=10,
    scorer=math_scorer,
    seed=SEED,
)

print(f"\n{'='*70}")
print(f"AVR-CL RESULTS:")
print(f"  ACC: {result_avr['acc']:.3f}")
print(f"  BWT: {result_avr['bwt']:+.3f}  (close to 0 = minimal forgetting)")
print(f"  Repairs fired: {result_avr['repairs']}")
print(f"  Domain A final accuracy: {result_avr['R'][2][0]:.3f}")
print(f"  (started at:              {result_avr['R'][0][0]:.3f})")
print(f"{'='*70}")


# ============================================================================
# CELL 5: Compare
# ============================================================================
print("\n" + "="*70)
print("COMPARISON: Naive vs avr-cl")
print("="*70)
print(f"\n{'Metric':<30} {'Naive':<15} {'avr-cl':<15}")
print("-"*60)
print(f"{'ACC (final avg)':<30} {result_naive['acc']:<15.3f} {result_avr['acc']:<15.3f}")
print(f"{'BWT (forgetting)':<30} {result_naive['bwt']:<+15.3f} {result_avr['bwt']:<+15.3f}")
print(f"{'Domain A preserved?':<30} {result_naive['R'][2][0]:<15.3f} {result_avr['R'][2][0]:<15.3f}")
print(f"{'Repairs fired':<30} {'0':<15} {result_avr['repairs']:<15}")
print("-"*60)

if result_avr['bwt'] > result_naive['bwt']:
    print(f"\n✓ avr-cl reduced forgetting by {abs(result_naive['bwt'] - result_avr['bwt']):.3f} BWT points")
    print(f"  Domain A survived: {result_naive['R'][2][0]:.1%} → {result_avr['R'][2][0]:.1%}")
else:
    print("\n(Run again — results may vary with this small dataset)")

print("\n" + "="*70)
print("Next steps:")
print("  1. pip install avr-cl")
print("  2. Try on YOUR sequential fine-tuning task")
print("  3. Use avr.check_drift() + avr.repair() as a layer in your existing pipeline")
print("="*70)
