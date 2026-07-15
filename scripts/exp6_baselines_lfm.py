"""
Experiment 6: Baselines on LFM2.5-1.2B-Instruct — Naive vs AVR vs EWC

Same math stream (GSM8K→MATH→AQuA→SVAMP), 500 examples/task.
3 conditions:
  - Naive: sequential SFT, no protection
  - AVR: PPL-gated repair (our method)
  - EWC: Elastic Weight Consolidation (Fisher-weighted L2 penalty)

Model: liquidai/LFM2.5-1.2B-Instruct (LiquidAI hybrid conv+attention, 1.2B)
  - LFM2.5 series: newer, more capable, instruct-tuned
  - Same Lfm2ForCausalLM architecture as LFM2 (10 conv + 6 attention layers)
  - Clean tokenizer (PreTrainedTokenizerFast), chat_template.jinja present
Download: ModelScope -> HF mirror -> direct HF (auto-fallback via _bootstrap)
Datasets: GitHub raw (no HuggingFace)
Package: pip-git -> git-clone -> raw-inline (auto-fallback via _bootstrap)

LFM2-specific notes:
  - attention output projection is "out_proj", NOT "o_proj"
  - needs transformers >= 4.57 (LFM2.5 config says 4.57.2)
  - use attn_implementation="sdpa" (T4 doesn't support flash_attention_2)
  - LoRA targets include conv layers (in_proj, out_proj) + attn (q/k/v/out_proj)
"""
# ============================================================================
# BOOTSTRAP — robust install + model download (see scripts/_bootstrap.py)
# ============================================================================
import os, sys, tempfile, urllib.request
try:
    # If running from a clone of the repo, _bootstrap.py is right here.
    from _bootstrap import (install_deps, install_avr, download_model,
                            patch_transformers_torch26, OUTPUT_DIR, DATA_CACHE)
except ImportError:
    # Kaggle workflow: paste this script alone. Auto-fetch _bootstrap.py.
    _src = "https://raw.githubusercontent.com/ARYAN2302/tiny-cl/main/scripts/_bootstrap.py"
    _dst = os.path.join(tempfile.gettempdir(), "_bootstrap.py")
    print(f"[exp6] fetching _bootstrap.py from {_src}", flush=True)
    urllib.request.urlretrieve(_src, _dst)
    sys.path.insert(0, tempfile.gettempdir())
    from _bootstrap import (install_deps, install_avr, download_model,
                            patch_transformers_torch26, OUTPUT_DIR, DATA_CACHE)

# LFM2.5 needs transformers >= 4.57
install_deps(transformers_pin=">=4.57.0,<5.0.0")
install_avr()                        # 3 fallback strategies for the avr package
patch_transformers_torch26()        # work around torch>=2.6 + transformers check

# ============================================================================
# Imports
# ============================================================================
import avr
from avr.repair import get_lora_state, set_lora_state
import re, random, math, gc, json, time, copy, shutil
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

SEED = 42

# LFM2.5-1.2B-Instruct — newer, more capable, instruct-tuned
# Same Lfm2ForCausalLM architecture as LFM2 (hybrid conv+attention)
# Clean tokenizer (PreTrainedTokenizerFast), chat_template.jinja present
# 10 conv + 6 attention layers, out_proj (not o_proj), in_proj (conv), w1/w3 (MLP)
MODEL_ID = "liquidai/LFM2.5-1.2B-Instruct"
# LFM2 uses "out_proj" (not "o_proj") for attention output.
# Also target conv layers (in_proj, out_proj) and MLP (w1, w3).
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w3"]

# ============================================================================
# Data loaders (identical to exp5 — GitHub raw)
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
    req = urllib.request.Request(url, headers={"User-Agent": "avr-cl/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return str(dest)

def load_gsm8k(n, split="train"):
    fn = "test" if split == "test" else "train"
    path = _download(f"{GSM8K_BASE}/{fn}.jsonl", DATA_CACHE / f"gsm8k_{fn}.jsonl")
    rows = [json.loads(l) for l in open(path)]
    rng = random.Random(SEED); rng.shuffle(rows)
    pairs = []
    for ex in rows[:n]:
        q, a = ex["question"], ex["answer"]
        m = re.search(r'####\s*(-?[\d,.]+)', a)
        gold = m.group(1).replace(",", "").strip() if m else a.strip()
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{q}", a, gold))
    return pairs

def load_math(n, split="train"):
    path = _download(MATH_URL, DATA_CACHE / "math_full.json")
    rows = json.load(open(path))
    rng = random.Random(SEED); rng.shuffle(rows)
    if split == "test": rows = rows[500:600]
    else: rows = rows[:n]
    pairs = []
    for ex in rows:
        q, sol = ex["problem"], ex["solution"]
        gold = ex.get("answer", "").strip()
        if not gold:
            m = re.findall(r'\\boxed\{([^}]+)\}', sol)
            gold = m[-1].strip() if m else ""
        if not gold:
            nums = re.findall(r'-?\d[\d.]*', sol)
            gold = nums[-1] if nums else ""
        pairs.append((f"Solve. End with \\boxed{{answer}}.\n\n{q}", sol, gold))
    return pairs if split == "test" else pairs[:n]

def load_aqua(n, split="train"):
    fn = "dev" if split == "test" else "train"
    path = _download(f"{AQUA_BASE}/{fn}.json", DATA_CACHE / f"aqua_{fn}.json")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = random.Random(SEED); rng.shuffle(rows)
    letters = ["A", "B", "C", "D", "E"]
    pairs = []
    for ex in rows[:n]:
        q, opts, correct = ex["question"], ex["options"], ex["correct"]
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
    path = _download(SVAMP_URL, DATA_CACHE / "svamp.json")
    rows = json.load(open(path))
    rng = random.Random(SEED); rng.shuffle(rows)
    if split == "test": rows = rows[500:600]
    else: rows = rows[:n]
    pairs = []
    for ex in rows:
        body, question = ex.get("Body", ""), ex.get("Question", "")
        answer, equation = ex.get("Answer", ""), ex.get("Equation", "")
        full_q = f"{body} {question}".strip()
        try:
            gf = float(answer)
            gold = str(int(gf)) if gf == int(gf) else str(gf)
        except: gold = str(answer)
        pairs.append((f"Solve step by step. End with #### <number>.\n\n{full_q}",
                      f"Equation: {equation}\n#### {gold}", gold))
    return pairs if split == "test" else pairs[:n]

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
# EWC BASELINE (identical to exp5)
# ============================================================================
class _TextDataset(Dataset):
    def __init__(self, token_ids, ctx_len):
        self.token_ids = token_ids
        self.ctx_len = ctx_len
        self.n_chunks = max(1, len(token_ids) // ctx_len)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.ctx_len
        chunk = self.token_ids[s:s+self.ctx_len]
        return {"input_ids": chunk, "labels": chunk.clone()}

def compute_fisher(model, tokenizer, examples, ctx_len, device, max_batches=50):
    from avr.model import format_example
    all_tokens = []
    for q, a, g in examples:
        all_tokens.extend(tokenizer.encode(format_example(tokenizer, q, a), add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = _TextDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    fisher = {n: torch.zeros_like(p.data) for n, p in model.named_parameters() if "lora_" in n}
    model.eval()
    n_batches = 0
    for batch in loader:
        if n_batches >= max_batches: break
        model.zero_grad()
        out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
        out.loss.backward()
        for n, p in model.named_parameters():
            if "lora_" in n and p.grad is not None:
                fisher[n] += p.grad.data ** 2
        n_batches += 1
    for n in fisher: fisher[n] /= max(n_batches, 1)
    model.zero_grad()
    return fisher

def train_sft_ewc(model, tokenizer, examples, fisher, opt_params, ewc_lambda=100.0,
                  epochs=3, lr=2e-4, batch_size=4, grad_accum=4, ctx_len=512, device="cuda", tag="ewc"):
    from avr.model import format_example
    all_tokens = []
    for q, a, g in examples:
        all_tokens.extend(tokenizer.encode(format_example(tokenizer, q, a), add_special_tokens=False))
    token_ids = torch.tensor(all_tokens, dtype=torch.long)
    dataset = _TextDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device), labels=batch["labels"].to(device))
            loss = out.loss
            if fisher and opt_params:
                ewc_pen = 0.0
                for n, p in model.named_parameters():
                    if "lora_" in n and n in fisher:
                        ewc_pen = ewc_pen + (fisher[n].to(device) * (p.data - opt_params[n].to(device)) ** 2).sum()
                loss = loss + (ewc_lambda / 2.0) * ewc_pen
            (loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(); accum = 0
                gs += 1; tl += out.loss.item()
                if gs % 50 == 0:
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)

def run_ewc(model_path, tasks_data, lora_rank, lora_targets, epochs, batch_size,
            grad_accum, ctx_len, scorer, seed, device="cuda"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    task_order = list(tasks_data.keys())
    T = len(task_order)
    from avr.model import load_model
    from avr.eval import evaluate
    print(f"\n{'='*70}", flush=True)
    print(f"EWC baseline | Model: {model_path} | Tasks: {task_order}", flush=True)
    print(f"LoRA r={lora_rank} | EWC λ=100 | Seed: {seed}", flush=True)
    print(f"{'='*70}", flush=True)
    model_obj, tokenizer = load_model(model_path, lora_rank, 128, lora_targets, device)
    R = [[0.0]*T for _ in range(T)]
    fisher, opt_params = {}, {}
    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task}\n{'='*60}", flush=True)
        train_ex = tasks_data[task]["train"]
        train_sft_ewc(model_obj, tokenizer, train_ex, fisher, opt_params,
                      ewc_lambda=100.0, epochs=epochs, lr=2e-4,
                      batch_size=batch_size, grad_accum=grad_accum,
                      ctx_len=ctx_len, device=device, tag="ewc")
        print(f"  Computing Fisher...", flush=True)
        fisher = compute_fisher(model_obj, tokenizer, train_ex, ctx_len, device)
        opt_params = get_lora_state(model_obj)
        print(f"  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate(model_obj, tokenizer, tasks_data[task_order[j]]["eval"],
                               task_order[j], scorer=scorer, device=device)
        print(f"  R[{ti}] = {[f'{v:.2f}' for v in R[ti]]}", flush=True)
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    acc = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt = float(np.mean([R[T-1][j] - R[j][j] for j in range(T-1)])) if T > 1 else 0.0
    ff = float(np.mean([max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)])) if T > 1 else 0.0
    del model_obj; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"acc": acc, "bwt": bwt, "ff": ff, "R": R, "repairs": 0, "task_order": task_order}

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("EXP 6: Baselines on LFM2.5-1.2B-Instruct", flush=True)
print("Naive vs AVR vs EWC", flush=True)
print("="*70, flush=True)

print("\nLoading data...", flush=True)
tasks_data = {
    "gsm8k": {"train": load_gsm8k(500), "eval": load_gsm8k(100, "test")},
    "math":  {"train": load_math(500),  "eval": load_math(100, "test")},
    "aqua":  {"train": load_aqua(500),  "eval": load_aqua(100, "test")},
    "svamp": {"train": load_svamp(500), "eval": load_svamp(100, "test")},
}
tasks_list = [(k, tasks_data[k]["train"], tasks_data[k]["eval"]) for k in tasks_data]

print(f"\nDownloading {MODEL_ID} (ModelScope -> HF mirror -> direct HF)...", flush=True)
MODEL_PATH = download_model(MODEL_ID, cache_dir=OUTPUT_DIR / "model_cache")
print(f"  Cached: {MODEL_PATH}", flush=True)

results = {}
COMMON = dict(
    model=MODEL_PATH, tasks=tasks_list,
    lora_rank=128, lora_targets=LORA_TARGETS,
    epochs=3, batch_size=4, grad_accum=4, ctx_len=512,
    scorer=math_scorer, seed=SEED,
)

# Condition 1: Naive
print(f"\n{'#'*60}\n# Naive (no protection)\n{'#'*60}", flush=True)
try:
    r = avr.run(**{**COMMON, "drift_threshold": 999.0, "repair_alpha": 0.0, "max_repair_steps": 0})
    results["naive"] = r
    print(f"  Naive: BWT={r['bwt']:+.3f} ACC={r['acc']:.3f}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    results["naive"] = {"error": str(e)}
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Condition 2: AVR
print(f"\n{'#'*60}\n# AVR (PPL-gated repair)\n{'#'*60}", flush=True)
try:
    r = avr.run(**{**COMMON, "drift_threshold": 1.15, "repair_alpha": 0.1, "max_repair_steps": 10})
    results["avr"] = r
    print(f"  AVR: BWT={r['bwt']:+.3f} ACC={r['acc']:.3f} Repairs={r['repairs']}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    results["avr"] = {"error": str(e)}
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Condition 3: EWC
print(f"\n{'#'*60}\n# EWC (Fisher-weighted L2 penalty)\n{'#'*60}", flush=True)
try:
    r = run_ewc(MODEL_PATH, tasks_data, lora_rank=128, lora_targets=LORA_TARGETS,
                epochs=3, batch_size=4, grad_accum=4, ctx_len=512,
                scorer=math_scorer, seed=SEED)
    results["ewc"] = r
    print(f"  EWC: BWT={r['bwt']:+.3f} ACC={r['acc']:.3f}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    results["ewc"] = {"error": str(e)}
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# Summary
print(f"\n{'='*70}", flush=True)
print("BASELINE COMPARISON — LFM2.5-1.2B-Instruct", flush=True)
print(f"{'='*70}", flush=True)
print(f"\n{'Method':<15} {'BWT':<10} {'ACC':<10} {'Repairs':<10}", flush=True)
print("-"*45, flush=True)
for cond, label in [("naive","Naive"), ("avr","AVR"), ("ewc","EWC")]:
    r = results.get(cond, {})
    if "bwt" in r:
        print(f"{label:<15} {r['bwt']:<+10.3f} {r['acc']:<10.3f} {r.get('repairs',0):<10}", flush=True)
    else:
        print(f"{label:<15} FAIL", flush=True)
print("-"*45, flush=True)

def strip(r):
    if "error" in r: return r
    return {k: v for k, v in r.items() if k != "R"}
with open(OUTPUT_DIR / "exp6_baselines_lfm.json", "w") as f:
    json.dump({k: strip(v) for k, v in results.items()}, f, indent=2, default=str)
print(f"\nSaved: {OUTPUT_DIR}/exp6_baselines_lfm.json", flush=True)
