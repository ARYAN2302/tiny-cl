"""
Test 2: Does AVR beat the thing practitioners actually do?

Practitioners mix 5-20% general instruction data into domain SFT to prevent
forgetting. This is the real competing solution — not naive SFT.

Conditions:
  A. Naive domain-SFT (no mitigation)
  B. Domain-SFT + 10% general data mixed in (practitioner baseline)
  C. AVR on domain-SFT (no mixed data, but repair after)

If AVR beats or matches B without needing old data, that's the real claim.

Setup:
  - Model: Qwen3-1.7B (or LFM2.5-230M on Kaggle)
  - Stream: general chat → math → general chat probe
    Task 1: Alpaca-style instruction following (general capability)
    Task 2: GSM8K math (domain fine-tune)
    Probe: Alpaca eval after task 2 — did the model forget how to follow instructions?

The probe tests whether math fine-tuning broke general instruction-following.
That's the 95% pain: "I fine-tuned on math and now my model can't chat."

Run on Kaggle T4 (~2 hours) or Modal A100 (~30 min).
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

import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False

import avr
import json, random, re, torch, gc, time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
random.seed(SEED)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_alpaca(n=500):
    """General instruction-following data (Alpaca format)."""
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
        pairs.append((prompt, output, output[:50]))  # gold = first 50 chars of output
    return pairs

def load_gsm8k(n=2000):
    """Math reasoning data."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(SEED)
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        q = ex["question"]; a = ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        prompt = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{q}"
        pairs.append((prompt, a, gold))
    return pairs

def load_alpaca_eval(n=100):
    """Alpaca eval set — different from train."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(SEED + 1)  # different seed for eval
    all_ex = list(ds); rng.shuffle(all_ex)
    pairs = []
    for ex in all_ex[:n]:
        instruction = ex["instruction"]
        inp = ex.get("input", "")
        output = ex["output"]
        prompt = f"{instruction}\n{inp}\n\nAnswer:" if inp else f"{instruction}\n\nAnswer:"
        pairs.append((prompt, output, output[:50]))
    return pairs

# ============================================================================
# CUSTOM SCORER — instruction following (substring match on key words)
# ============================================================================
def instruction_scorer(response, gold):
    """Check if the response is a reasonable instruction-following output.
    Simple heuristic: non-empty, >10 chars, not just repeating the prompt."""
    resp = response.strip()
    if len(resp) < 10:
        return 0.0
    # Check if it's just echoing the prompt
    if resp.lower() == gold.lower()[:len(resp)]:
        return 0.0
    return 1.0

# ============================================================================
# RUN CONDITIONS
# ============================================================================
def run_condition_a(model_id, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval):
    """Condition A: Naive — SFT on alpaca, then SFT on GSM8K. No mitigation."""
    print(f"\n{'#'*70}\n# CONDITION A: Naive (no mitigation)\n{'#'*70}", flush=True)
    result = avr.run(
        model=model_id,
        tasks=[
            ("alpaca", alpaca_train, alpaca_eval),
            ("gsm8k", gsm8k_train, gsm8k_eval),
        ],
        lora_rank=128,
        epochs=3,
        batch_size=4,
        grad_accum=4,
        ctx_len=512,
        drift_threshold=1.15,
        repair_alpha=0.1,
        max_repair_steps=10,
        two_stream=False,
        scorer=None,  # default scorer for both tasks
        seed=SEED,
    )
    return result

def run_condition_b(model_id, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval):
    """Condition B: Practitioner baseline — mix 10% alpaca into GSM8K training."""
    print(f"\n{'#'*70}\n# CONDITION B: 10% data mixing (practitioner baseline)\n{'#'*70}", flush=True)
    # Mix 10% alpaca into gsm8k train
    n_mix = len(gsm8k_train) // 10
    rng = random.Random(SEED)
    alpaca_mix = alpaca_train[:n_mix]
    gsm8k_mixed = list(gsm8k_train) + alpaca_mix
    rng.shuffle(gsm8k_mixed)

    result = avr.run(
        model=model_id,
        tasks=[
            ("alpaca", alpaca_train, alpaca_eval),
            ("gsm8k_mixed", gsm8k_mixed, gsm8k_eval),
        ],
        lora_rank=128,
        epochs=3,
        batch_size=4,
        grad_accum=4,
        ctx_len=512,
        # No AVR — this is the baseline
        drift_threshold=999.0,  # never fire repair
        repair_alpha=0.0,
        max_repair_steps=0,
        two_stream=False,
        seed=SEED,
    )
    return result

def run_condition_c(model_id, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval):
    """Condition C: AVR — SFT on alpaca, then SFT on GSM8K with drift detect + repair."""
    print(f"\n{'#'*70}\n# CONDITION C: AVR (drift detect + repair, no mixed data)\n{'#'*70}", flush=True)
    result = avr.run(
        model=model_id,
        tasks=[
            ("alpaca", alpaca_train, alpaca_eval),
            ("gsm8k", gsm8k_train, gsm8k_eval),
        ],
        lora_rank=128,
        epochs=3,
        batch_size=4,
        grad_accum=4,
        ctx_len=512,
        drift_threshold=1.15,
        repair_alpha=0.1,
        max_repair_steps=10,
        two_stream=False,
        seed=SEED,
    )
    return result

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("TEST 2: AVR vs data mixing vs naive", flush=True)
print(f"Model: {MODEL_ID} | Seed: {SEED}", flush=True)
print(f"Stream: Alpaca (general) → GSM8K (math domain)", flush=True)
print(f"Probe: Alpaca accuracy after math fine-tune", flush=True)
print(f"  A. Naive (no mitigation)", flush=True)
print(f"  B. 10% data mixing (practitioner baseline)", flush=True)
print(f"  C. AVR (drift detect + repair, no old data)", flush=True)
print("="*70, flush=True)

# Load data
print("\nLoading data...", flush=True)
alpaca_train = load_alpaca(n=500)
gsm8k_train = load_gsm8k(n=2000)
alpaca_eval = load_alpaca_eval(n=50)
# GSM8K eval: use test set
from datasets import load_dataset
ds_te = load_dataset("openai/gsm8k", "main", split="test")
rng = random.Random(SEED)
all_te = list(ds_te); rng.shuffle(all_te)
gsm8k_eval = []
for ex in all_te[:50]:
    q = ex["question"]; a = ex["answer"]
    m = re.search(r'####\s*(-?[\d,.]+)', a)
    gold = m.group(1).replace(",", "").strip() if m else a.strip()
    prompt = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{q}"
    gsm8k_eval.append((prompt, a, gold))

print(f"  Alpaca: {len(alpaca_train)} train, {len(alpaca_eval)} eval", flush=True)
print(f"  GSM8K: {len(gsm8k_train)} train, {len(gsm8k_eval)} eval", flush=True)

# Run all 3 conditions
results = {}

try:
    results["A_naive"] = run_condition_a(MODEL_ID, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval)
    # Alpaca accuracy after GSM8K = R[1][0] (task 2 row, task 1 col)
    print(f"\n  A: Alpaca after GSM8K = {results['A_naive']['R'][1][0]:.3f}", flush=True)
except Exception as e:
    print(f"  A failed: {e}", flush=True)
    results["A_naive"] = {"error": str(e)}

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

try:
    results["B_mixing"] = run_condition_b(MODEL_ID, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval)
    print(f"\n  B: Alpaca after GSM8K = {results['B_mixing']['R'][1][0]:.3f}", flush=True)
except Exception as e:
    print(f"  B failed: {e}", flush=True)
    results["B_mixing"] = {"error": str(e)}

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

try:
    results["C_avr"] = run_condition_c(MODEL_ID, alpaca_train, gsm8k_train, alpaca_eval, gsm8k_eval)
    print(f"\n  C: Alpaca after GSM8K = {results['C_avr']['R'][1][0]:.3f}", flush=True)
except Exception as e:
    print(f"  C failed: {e}", flush=True)
    results["C_avr"] = {"error": str(e)}

# Summary
print(f"\n{'='*70}", flush=True)
print("TEST 2 RESULTS", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n{'Condition':<35} {'Alpaca after GSM8K':<20} {'BWT':<10} {'Repairs':<10}", flush=True)
print("-"*75, flush=True)
for cond, label in [("A_naive", "A: Naive (no mitigation)"),
                     ("B_mixing", "B: 10% data mixing"),
                     ("C_avr", "C: AVR (no old data)")]:
    if cond in results and "R" in results[cond]:
        r = results[cond]
        alpaca_acc = r["R"][1][0] if len(r["R"]) > 1 else 0
        print(f"{label:<35} {alpaca_acc:<20.3f} {r['bwt']:<+10.3f} {r['repairs']:<10}", flush=True)
    else:
        print(f"{label:<35} ERROR", flush=True)
print("-"*75, flush=True)

# The real question: does C beat or match B?
if "R" in results.get("C_avr", {}) and "R" in results.get("B_mixing", {}):
    c_alpaca = results["C_avr"]["R"][1][0]
    b_alpaca = results["B_mixing"]["R"][1][0]
    a_alpaca = results["A_naive"]["R"][1][0] if "R" in results.get("A_naive", {}) else 0
    print(f"\n  Verdict:", flush=True)
    print(f"  Naive:  Alpaca = {a_alpaca:.3f}", flush=True)
    print(f"  Mixing: Alpaca = {b_alpaca:.3f} (practitioner baseline, needs old data)", flush=True)
    print(f"  AVR:    Alpaca = {c_alpaca:.3f} (no old data, just repair)", flush=True)
    if c_alpaca >= b_alpaca:
        print(f"\n  ✅ AVR matches or beats data mixing WITHOUT needing old data.", flush=True)
    elif c_alpaca > a_alpaca:
        print(f"\n  ◐ AVR beats naive but doesn't match data mixing.", flush=True)
    else:
        print(f"\n  ❌ AVR doesn't help in this setup.", flush=True)

with open(OUTPUT_DIR / "test2_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "R"} or v
               for k, v in results.items()}, f, indent=2, default=str)
print(f"\nResults saved: {OUTPUT_DIR}/test2_results.json", flush=True)
print(f"\nDONE.", flush=True)
