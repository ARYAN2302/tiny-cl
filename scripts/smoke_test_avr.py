"""
Smoke test: does avr.run() work end-to-end?

Tiny model, tiny data, 2 tasks, just verify the loop runs and produces numbers.
Run on Kaggle T4 — should take ~5 minutes.
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess, sys
# Install deps + the avr package itself from the repo
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.13.0", "datasets>=3.0.0", "accelerate>=1.0.0",
    "sentencepiece", "protobuf", "packaging"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/ARYAN2302/tiny-cl.git#subdirectory=avr"], check=True)

# Patch transformers dynamo flag (Kaggle numpy compat)
import transformers.utils.import_utils as _iu
_iu._is_torch_greater_or_equal_than_2_6 = False
_iu.is_torch_greater_or_equal_than_2_6 = lambda: False

import avr
import random, json

random.seed(42)

# Tiny synthetic data — 20 examples per task, 5 eval per task
# (question, answer, gold) tuples
def make_tiny_data(n_train=20, n_eval=5):
    """Two tiny tasks: addition and subtraction."""
    train_a, eval_a = [], []
    train_b, eval_b = [], []
    for i in range(n_train + n_eval):
        a, b = random.randint(1, 50), random.randint(1, 50)
        if i < n_train:
            train_a.append((f"What is {a} + {b}?", f"{a} + {b} = {a+b}. #### {a+b}", str(a+b)))
            train_b.append((f"What is {a} - {b}?", f"{a} - {b} = {a-b}. #### {a-b}", str(a-b)))
        else:
            eval_a.append((f"What is {a} + {b}?", f"{a} + {b} = {a+b}. #### {a+b}", str(a+b)))
            eval_b.append((f"What is {a} - {b}?", f"{a} - {b} = {a-b}. #### {a-b}", str(a-b)))
    return train_a, eval_a, train_b, eval_b

train_a, eval_a, train_b, eval_b = make_tiny_data()

print("="*60, flush=True)
print("SMOKE TEST: avr.run() on tiny synthetic data", flush=True)
print("="*60, flush=True)

try:
    result = avr.run(
        model="LiquidAI/LFM2.5-230M",
        tasks=[
            ("addition", train_a, eval_a),
            ("subtraction", train_b, eval_b),
        ],
        lora_rank=32,
        epochs=1,           # just 1 epoch for smoke test
        batch_size=4,
        grad_accum=2,
        ctx_len=256,        # tiny context for tiny data
        max_repair_steps=5,
        seed=42,
    )

    print(f"\n{'='*60}", flush=True)
    print("SMOKE TEST PASSED", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  ACC: {result['acc']:.3f}", flush=True)
    print(f"  BWT: {result['bwt']:+.3f}", flush=True)
    print(f"  FF:  {result['ff']:.3f}", flush=True)
    print(f"  Repairs: {result['repairs']}", flush=True)
    print(f"  R-matrix: {result['R']}", flush=True)

except Exception as e:
    import traceback
    print(f"\nSMOKE TEST FAILED: {e}", flush=True)
    traceback.print_exc()
    print(f"\n{'='*60}", flush=True)
    print("SMOKE TEST FAILED", flush=True)
    print(f"{'='*60}", flush=True)
