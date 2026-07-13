"""
Experiment 3: 8-Task TRACE, AVR-only (no Two-Stream).

Tests whether AVR alone scales to 8 tasks without the Two-Stream crutch.
Published LoRA CL methods on TRACE 8-task are ALL strongly negative
(best: GORP at -0.7 on 7B). If AVR achieves >= -0.05, that's first-in-class.

Uses the avr package.
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "datasets>=3.0.0", "accelerate>=1.0.0",
    "sentencepiece", "protobuf", "packaging", "gdown"], check=True)
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

# ============================================================================
# TRACE DATA — download from Google Drive, 0.5K variant
# ============================================================================
TRACE_VARIANT = "LLM-CL-Benchmark_5000"
TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
TRACE_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
               "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

def load_trace():
    import gdown, zipfile
    trace_dir = OUTPUT_DIR / "trace_data"
    if not (trace_dir.exists() and any(trace_dir.iterdir())):
        for d in OUTPUT_DIR.rglob(f"*{TRACE_VARIANT}*"):
            if d.is_dir():
                trace_dir = d; break
        else:
            print("  Downloading TRACE...", flush=True)
            zip_path = OUTPUT_DIR / "trace_benchmark.zip"
            gdown.download(id=TRACE_GDRIVE_ID, output=str(zip_path), quiet=False)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(OUTPUT_DIR)
            for d in OUTPUT_DIR.rglob(f"*{TRACE_VARIANT}*"):
                if d.is_dir():
                    trace_dir = d; break

    rng = random.Random(SEED)
    tasks = []
    for task_name in TRACE_TASKS:
        task_dir = trace_dir / task_name
        with open(task_dir / "train.json") as f: train_data = json.load(f)
        with open(task_dir / "test.json") as f: test_data = json.load(f)
        train_pairs = [(ex["prompt"], ex["answer"], ex["answer"]) for ex in train_data]
        test_pairs = [(ex["prompt"], ex["answer"], ex["answer"]) for ex in test_data]
        rng.shuffle(train_pairs); rng.shuffle(test_pairs)
        train_pairs = train_pairs[:500]  # 0.5K variant
        test_pairs = test_pairs[:100]
        tasks.append((task_name, train_pairs, test_pairs))
        print(f"  {task_name}: {len(train_pairs)} train, {len(test_pairs)} eval", flush=True)
    return tasks

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("EXP 3: 8-Task TRACE, AVR-only (no Two-Stream)", flush=True)
print("="*70, flush=True)

print("\nLoading TRACE 0.5K...", flush=True)
tasks = load_trace()

# Condition A: Naive
print(f"\n{'#'*60}\n# Condition A: Naive (8 tasks)\n{'#'*60}", flush=True)
try:
    result_naive = avr.run(
        model="Qwen/Qwen3-1.7B",
        tasks=tasks,
        lora_rank=32,
        lora_alpha=32,
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj"],
        epochs=3,
        lr=2e-4,
        batch_size=4,
        grad_accum=4,
        ctx_len=512,
        drift_threshold=999.0,  # never fire repair
        repair_alpha=0.0,
        max_repair_steps=0,
        two_stream=False,
        seed=SEED,
    )
    print(f"\n  Naive: ACC={result_naive['acc']:.3f} BWT={result_naive['bwt']:+.3f} Repairs={result_naive['repairs']}", flush=True)
except Exception as e:
    print(f"  Naive failed: {e}", flush=True)
    import traceback; traceback.print_exc()
    result_naive = {"error": str(e)}

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Condition B: AVR-only
print(f"\n{'#'*60}\n# Condition B: AVR-only (8 tasks)\n{'#'*60}", flush=True)
try:
    result_avr = avr.run(
        model="Qwen/Qwen3-1.7B",
        tasks=tasks,
        lora_rank=32,
        lora_alpha=32,
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
        seed=SEED,
    )
    print(f"\n  AVR: ACC={result_avr['acc']:.3f} BWT={result_avr['bwt']:+.3f} Repairs={result_avr['repairs']}", flush=True)
except Exception as e:
    print(f"  AVR failed: {e}", flush=True)
    import traceback; traceback.print_exc()
    result_avr = {"error": str(e)}

# Summary
print(f"\n{'='*70}", flush=True)
print("8-TASK TRACE RESULTS (AVR-only vs Naive)", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n{'Method':<25} {'ACC':<10} {'BWT':<10} {'FF':<10} {'Repairs':<10}", flush=True)
print("-"*65, flush=True)
if "bwt" in result_naive:
    print(f"{'Naive':<25} {result_naive['acc']:<10.3f} {result_naive['bwt']:<+10.3f} {result_naive['ff']:<10.3f} {'-':<10}", flush=True)
if "bwt" in result_avr:
    print(f"{'AVR-only':<25} {result_avr['acc']:<10.3f} {result_avr['bwt']:<+10.3f} {result_avr['ff']:<10.3f} {result_avr['repairs']:<10}", flush=True)
print("-"*65, flush=True)
print(f"\nPublished TRACE 8-task LoRA baselines (7B, for reference):", flush=True)
print(f"  GORP (ACL 2025):     BWT = -0.7", flush=True)
print(f"  O-LoRA:              BWT = -4.3", flush=True)
print(f"  CoDyRA (2025):       BWT = -3.25", flush=True)

with open(OUTPUT_DIR / "exp3_trace_8task.json", "w") as f:
    json.dump({"naive": result_naive, "avr": result_avr}, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp3_trace_8task.json", flush=True)
