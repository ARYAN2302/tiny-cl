# Experiment 3: 8-Task TRACE - Naive vs AVR
# Self-contained: bootstrap+avr+TRACE inlined. Only needs PyPI + ModelScope.
# TRACE tasks: C-STANCE, FOMC, MeetingBank, Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten
# ===========================================================================
# BOOTSTRAP — inlined from scripts/_bootstrap.py (self-contained, no fetch)
# ===========================================================================
import os
import sys
import re
import subprocess
import urllib.request
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Env tweaks — must be set before any HF/torch import.
# HF_HUB_DISABLE_XET=1 is the key fix for the hf-xet 403 errors that forced
# the switch to ModelScope in the first place. With xet disabled, regular
# HTTPS downloads from HF work fine from Kaggle.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYDEVD_DISABLE_FILE_VALIDATION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Output dirs — Kaggle /kaggle/working if present, else ./output
OUTPUT_DIR = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE = OUTPUT_DIR / "data_cache"
DATA_CACHE.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/ARYAN2302/tiny-cl.git"
REPO_RAW = "https://raw.githubusercontent.com/ARYAN2302/tiny-cl/main"


# ---------------------------------------------------------------------------
# 1. Dependency install
# ---------------------------------------------------------------------------
def _pip(*args, check=True, timeout=600, quiet=True):
    """Run pip with a timeout. On failure, print captured stderr before raising."""
    flags = ["-q"] if quiet else []
    cmd = [sys.executable, "-m", "pip", "install", *flags, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 and check:
        # Surface the actual pip error — the -q flag hides it by default
        print(f"[bootstrap] pip FAILED: {' '.join(args)}", flush=True)
        print(f"[bootstrap] pip returncode: {r.returncode}", flush=True)
        if r.stderr:
            # Print last 2000 chars of stderr (the useful part)
            print("[bootstrap] pip stderr (tail):\n" + r.stderr[-2000:], flush=True)
        if r.stdout:
            print("[bootstrap] pip stdout (tail):\n" + r.stdout[-1000:], flush=True)
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


def _pip_maybe(*args, timeout=600):
    """Run pip, return True on success, False on failure (no raise)."""
    try:
        r = _pip(*args, check=False, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[bootstrap] pip timed out: {' '.join(args)}", flush=True)
        return False


def install_deps(extra=None, transformers_pin=None):
    """
    Install common deps for exp5/exp6. Resilient to individual package failures.

    Strategy:
      1. Try installing all packages in one batch (fast path).
      2. If that fails, install them one at a time so a single bad package
         doesn't kill the whole run. Print a warning for each failure.
      3. modelscope is OPTIONAL — if it fails to install, we skip it and
         rely on the HF fallback in download_model(). Same for transformers_pin.

    Args:
        extra: list of extra pip specs to install.
        transformers_pin: e.g. ">=4.57.0,<5.0.0" for LFM2.5 support.
                         If None, installs whatever is already there (no force).
    """
    # Core packages — these are required and must all install.
    core_pkgs = [
        "peft>=0.13.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "packaging",
        "huggingface_hub>=0.26.0",
        "gdown",  # needed for HF_HUB_DISABLE_XET support
        "huggingface_hub>=0.26.0",  # needed for HF_HUB_DISABLE_XET support
    ]
    # Optional packages — if they fail, we continue
    # modelscope is optional because we download directly from ModelScope HTTP API
    optional_pkgs = ["modelscope"]
    if transformers_pin:
        optional_pkgs.append(f"transformers{transformers_pin}")
    if extra:
        optional_pkgs.extend(extra)

    print("[bootstrap] Installing deps...", flush=True)

    # Try batch install of core + optional first (fast path)
    all_pkgs = core_pkgs + optional_pkgs
    if _pip_maybe(*all_pkgs):
        print("[bootstrap] All deps installed in one batch.", flush=True)
    else:
        # Batch failed — install core one-by-one (required), then optional (best-effort)
        print("[bootstrap] Batch install failed, installing individually...", flush=True)
        for pkg in core_pkgs:
            if not _pip_maybe(pkg):
                # Core package failed — this is fatal, but try without version pin
                pkg_name = re.split(r"[><=!]", pkg, maxsplit=1)[0]
                print(f"[bootstrap] WARNING: {pkg} failed, trying {pkg_name} unpinned...", flush=True)
                if not _pip_maybe(pkg_name):
                    raise RuntimeError(f"Required package {pkg_name} failed to install")
        for pkg in optional_pkgs:
            if not _pip_maybe(pkg):
                pkg_name = re.split(r"[><=!]", pkg, maxsplit=1)[0]
                print(f"[bootstrap] WARNING: optional {pkg} failed ({pkg_name} unavailable). "
                      f"Will use fallbacks.", flush=True)

    # torchao crashes on T4; hf-xet is what causes the 403s. Kill both.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "hf-xet"],
                   check=False, capture_output=True)
    print("[bootstrap] Deps ready.", flush=True)


# ---------------------------------------------------------------------------
# 2. avr package — INLINED SOURCE (no github/network needed)
# ---------------------------------------------------------------------------
# The avr package source (~25KB, 8 files) is embedded directly below.
# At runtime, _inline_avr_package() writes it to /tmp/avr-inline/avr/ and
# adds that dir to sys.path. This avoids all dependency on github.com
# and raw.githubusercontent.com (which sometimes have DNS issues on Kaggle).
# The only network calls this script makes are:
#   - PyPI (pip install of deps below)
#   - ModelScope or HuggingFace (for model download)
# ---------------------------------------------------------------------------

AVR_SOURCES = {
    '__init__.py': '"""\navr-cl: continual post-training with drift detection + repair.\n\nLEARN → VERIFY → REPAIR. Each phase is a separate module you can swap:\n  - avr.learn: train_sft, consolidate (LEARN)\n  - avr.verify: compute_ppl, eval_ppls, check_drift (VERIFY)\n  - avr.repair: get_lora_state, set_lora_state, reset_lora, repair (REPAIR)\n  - avr.eval: evaluate, generate_batch, default_scorer (evaluation)\n  - avr.model: load_model, format_prompt, format_example (model handling)\n\nQuickstart:\n    import avr\n    result = avr.run(\n        model="Qwen/Qwen3-1.7B",\n        tasks=[\n            ("gsm8k", train_pairs, eval_pairs),\n            ("math", train_pairs, eval_pairs),\n        ],\n        lora_rank=128,\n    )\n    print(f"BWT: {result[\'bwt\']:+.3f}  Repairs: {result[\'repairs\']}")\n\nCustom repair operator (TIES, TaskArithmetic, etc.):\n    import avr\n\n    def my_repair(model, snapshot, alpha, device):\n        # your merge logic here\n        return n_params_touched\n\n    result = avr.run(model=..., tasks=..., repair_fn=my_repair)\n"""\nfrom .run import run, compute_metrics\nfrom .model import load_model, detect_lora_targets, format_prompt, format_example\nfrom .learn import train_sft, consolidate\nfrom .verify import compute_ppl, eval_ppls, check_drift\nfrom .repair import get_lora_state, set_lora_state, reset_lora, repair\nfrom .eval import evaluate, generate_batch, default_scorer, normalize_answer\n\n__version__ = "0.1.0"\n\n__all__ = [\n    "run",\n    "compute_metrics",\n    "load_model", "detect_lora_targets", "format_prompt", "format_example",\n    "train_sft", "consolidate",\n    "compute_ppl", "eval_ppls", "check_drift",\n    "get_lora_state", "set_lora_state", "reset_lora", "repair",\n    "evaluate", "generate_batch", "default_scorer", "normalize_answer",\n]\n',
    'model.py': '"""Model loading, LoRA setup, and chat template wrapping."""\nimport torch\nfrom typing import Optional\n\n\ndef load_model(model_id: str, lora_rank: int = 128, lora_alpha: int = 128,\n               lora_targets: list = None, device: str = "cuda"):\n    from transformers import AutoModelForCausalLM, AutoTokenizer\n    from peft import LoraConfig, get_peft_model, TaskType\n\n    print(f"  Loading {model_id}...", flush=True)\n    tokenizer = AutoTokenizer.from_pretrained(model_id)\n    if tokenizer.pad_token is None:\n        tokenizer.pad_token = tokenizer.eos_token\n\n    model = AutoModelForCausalLM.from_pretrained(\n        model_id, dtype=torch.bfloat16, device_map=device,\n        attn_implementation="sdpa")\n\n    if lora_targets is None:\n        lora_targets = detect_lora_targets(model)\n        print(f"  Auto-detected LoRA targets: {lora_targets}", flush=True)\n\n    lora_config = LoraConfig(\n        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,\n        target_modules=lora_targets, bias="none", task_type=TaskType.CAUSAL_LM)\n    model = get_peft_model(model, lora_config)\n    model.gradient_checkpointing_enable()\n    model.enable_input_require_grads()\n\n    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\n    total = sum(p.numel() for p in model.parameters())\n    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)\n    return model, tokenizer\n\n\ndef detect_lora_targets(model):\n    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",\n                  "gate_proj", "up_proj", "down_proj",\n                  "in_proj", "out_proj", "conv1d"]\n    found = []\n    for name, _ in model.named_modules():\n        for c in candidates:\n            if name.endswith(c) and c not in found:\n                found.append(c)\n    return found if found else ["q_proj", "v_proj"]\n\n\ndef format_prompt(tokenizer, question: str) -> str:\n    messages = [{"role": "user", "content": question}]\n    try:\n        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n    except Exception:\n        return f"User: {question}\\nAssistant:"\n\n\ndef format_example(tokenizer, question: str, answer: str) -> str:\n    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]\n    try:\n        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)\n    except Exception:\n        text = f"User: {question}\\nAssistant: {answer}"\n    return text + tokenizer.eos_token\n',
    'learn.py': '"""LEARN phase: SFT training + two-stream consolidation."""\nimport torch, time\nimport torch.nn.functional as F\nfrom torch.utils.data import Dataset, DataLoader\nfrom .model import format_example\nfrom .repair import get_lora_state, set_lora_state\n\n\nclass _TextDataset(Dataset):\n    def __init__(self, token_ids, ctx_len):\n        self.token_ids = token_ids\n        self.ctx_len = ctx_len\n        self.n_chunks = max(1, len(token_ids) // ctx_len)\n    def __len__(self): return self.n_chunks\n    def __getitem__(self, idx):\n        s = idx * self.ctx_len\n        e = s + self.ctx_len\n        chunk = self.token_ids[s:e]\n        return {"input_ids": chunk, "labels": chunk.clone()}\n\n\ndef train_sft(model, tokenizer, examples, epochs=3, lr=2e-4, batch_size=4,\n              grad_accum=4, ctx_len=512, device="cuda", tag="sft"):\n    all_tokens = []\n    for question, answer, gold in examples:\n        text = format_example(tokenizer, question, answer)\n        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))\n    token_ids = torch.tensor(all_tokens, dtype=torch.long)\n    dataset = _TextDataset(token_ids, ctx_len)\n    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)\n\n    for n, p in model.named_parameters():\n        if "lora_" in n: p.requires_grad = True\n        else: p.requires_grad = False\n    trainable = [p for p in model.parameters() if p.requires_grad]\n    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)\n    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)\n\n    gs, tl = 0, 0.0\n    t0 = time.time()\n    accum = 0\n    opt.zero_grad()\n    for epoch in range(epochs):\n        for batch in loader:\n            model.train()\n            out = model(input_ids=batch["input_ids"].to(device),\n                       labels=batch["labels"].to(device))\n            (out.loss / grad_accum).backward()\n            accum += 1\n            if accum >= grad_accum:\n                torch.nn.utils.clip_grad_norm_(trainable, 1.0)\n                opt.step()\n                opt.zero_grad()\n                accum = 0\n                gs += 1\n                tl += out.loss.item()\n                if gs % 50 == 0:\n                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)\n\n\ndef consolidate(model, tokenizer, hippo_state, neo_state, examples,\n                epochs=1, lr=1e-4, batch_size=4, grad_accum=4, ctx_len=512, device="cuda"):\n    print(f"    [consolid] KL distill ({epochs} epoch)", flush=True)\n    all_tokens = []\n    for question, answer, gold in examples:\n        text = format_example(tokenizer, question, answer)\n        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))\n    token_ids = torch.tensor(all_tokens, dtype=torch.long)\n    dataset = _TextDataset(token_ids, ctx_len)\n    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)\n\n    for n, p in model.named_parameters():\n        if "lora_" in n: p.requires_grad = True\n        else: p.requires_grad = False\n    trainable = [p for p in model.parameters() if p.requires_grad]\n    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)\n\n    gs, tl = 0, 0.0\n    t0 = time.time()\n    accum = 0\n    opt.zero_grad()\n    for epoch in range(epochs):\n        for batch in loader:\n            input_ids = batch["input_ids"].to(device)\n            set_lora_state(model, hippo_state, device)\n            model.eval()\n            with torch.no_grad():\n                hippo_logits = model(input_ids=input_ids).logits\n                p_hippo = F.softmax(hippo_logits[..., :-1, :].contiguous().float(), dim=-1)\n            del hippo_logits\n\n            set_lora_state(model, neo_state, device)\n            model.train()\n            neo_logits = model(input_ids=input_ids).logits\n            shift_neo = neo_logits[..., :-1, :].contiguous()\n            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)\n            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction=\'batchmean\', log_target=False)\n            (kl_loss / grad_accum).backward()\n            accum += 1\n            if accum >= grad_accum:\n                torch.nn.utils.clip_grad_norm_(trainable, 1.0)\n                opt.step()\n                opt.zero_grad()\n                accum = 0\n                gs += 1\n                tl += kl_loss.item()\n                if gs % 50 == 0:\n                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)\n            del neo_logits, log_p_neo, p_hippo, kl_loss, shift_neo\n            neo_state = get_lora_state(model)\n    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)\n    return neo_state\n',
    'verify.py': '"""VERIFY phase: PPL drift detection."""\nimport math\nimport torch\nfrom .model import format_example\n\n\ndef compute_ppl(model, tokenizer, examples, device="cuda", max_samples=100):\n    model.eval()\n    total_loss, total_tokens = 0.0, 0\n    for question, answer, gold in examples[:max_samples]:\n        text = format_example(tokenizer, question, answer)\n        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)\n        with torch.no_grad():\n            outputs = model(**inputs, labels=inputs["input_ids"])\n        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]\n        total_tokens += inputs["input_ids"].shape[1]\n    model.train()\n    return math.exp(total_loss / max(total_tokens, 1))\n\n\ndef eval_ppls(model, tokenizer, tasks_data, task_order, trained_so_far, device="cuda"):\n    ppls = {}\n    for i, task in enumerate(task_order):\n        if i >= trained_so_far:\n            break\n        ppls[task] = compute_ppl(model, tokenizer, tasks_data[task]["train"], device=device)\n    return ppls\n\n\ndef check_drift(current_ppls, best_ppls, completed_tasks, threshold=1.15):\n    drifted = {}\n    for task in completed_tasks:\n        if task not in current_ppls or task not in best_ppls:\n            continue\n        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0\n        if ratio > threshold:\n            drifted[task] = {"current": current_ppls[task], "best": best_ppls[task], "ratio": ratio}\n    return drifted\n',
    'repair.py': '"""REPAIR phase: LoRA state management + weight interpolation."""\nimport math\nimport torch\nimport torch.nn.init as init\n\n\ndef get_lora_state(model):\n    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}\n\ndef set_lora_state(model, state, device="cuda"):\n    for n, p in model.named_parameters():\n        if "lora_" in n and n in state:\n            p.data.copy_(state[n].to(device).to(p.data.dtype))\n\ndef reset_lora(model):\n    for n, p in model.named_parameters():\n        if "lora_A" in n:\n            init.kaiming_uniform_(p.data, a=math.sqrt(5))\n        elif "lora_B" in n:\n            p.data.zero_()\n\ndef repair(model, snapshot, alpha=0.1, device="cuda"):\n    n = 0\n    for name, p in model.named_parameters():\n        if "lora_" in name and name in snapshot:\n            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))\n            n += 1\n    return n\n',
    'eval.py': '"""Batched evaluation + scoring."""\nimport gc, re, time\nimport torch\nfrom .model import format_prompt\n\n\ndef generate_batch(model, tokenizer, questions, max_new_tokens=200, batch_size=8, device="cuda"):\n    results = []\n    gc_was = getattr(model, "gradient_checkpointing", False)\n    if gc_was:\n        try: model.gradient_checkpointing_disable()\n        except: pass\n    model.eval()\n    try:\n        for i in range(0, len(questions), batch_size):\n            batch = questions[i:i+batch_size]\n            texts = [format_prompt(tokenizer, q) for q in batch]\n            tokenizer.padding_side = "left"\n            inputs = tokenizer(texts, return_tensors="pt", truncation=True,\n                             max_length=1024, padding=True).to(device)\n            with torch.no_grad():\n                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,\n                    do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=1.0)\n            for out in outputs:\n                input_len = inputs["input_ids"].shape[1]\n                results.append(tokenizer.decode(out[input_len:], skip_special_tokens=True).strip())\n    finally:\n        if gc_was:\n            try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()\n            except: pass\n    return results\n\n\ndef normalize_answer(s):\n    s = s.strip().lower()\n    s = re.sub(r\'[^\\w\\s.-]\', \' \', s)\n    return \' \'.join(s.split())\n\n\ndef default_scorer(response, gold):\n    resp = normalize_answer(response)\n    g = normalize_answer(gold)\n    if resp == g: return 1.0\n    if g in resp or resp in g: return 1.0\n    g_spaces = g.replace(\'_\', \' \')\n    resp_spaces = resp.replace(\'_\', \' \')\n    if g_spaces in resp_spaces or resp_spaces in g_spaces: return 1.0\n    return 0.0\n\n\ndef evaluate(model, tokenizer, eval_examples, task_name, scorer=None,\n             max_questions=200, batch_size=8, device="cuda"):\n    if scorer is None:\n        scorer = default_scorer\n    total = min(len(eval_examples), max_questions)\n    examples = eval_examples[:total]\n    questions = [ex[0] for ex in examples]\n    golds = [ex[2] for ex in examples]\n    print(f"    Eval {task_name} ({total} Qs)...", flush=True)\n    correct = 0\n    t0 = time.time()\n    for i in range(0, len(questions), batch_size):\n        batch_q = questions[i:i+batch_size]\n        batch_g = golds[i:i+batch_size]\n        responses = generate_batch(model, tokenizer, batch_q, max_new_tokens=200,\n                                   batch_size=len(batch_q), device=device)\n        for r, g in zip(responses, batch_g):\n            if scorer(r, g): correct += 1\n    acc = correct / total\n    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)\n    if torch.cuda.is_available(): torch.cuda.empty_cache()\n    return acc\n',
    'run.py': '"""\navr.run — the orchestrator.\n\nWires LEARN → VERIFY → REPAIR across a task stream.\nEach phase is swappable: import from avr.learn, avr.verify, avr.repair\nand pass your own implementations if needed.\n"""\nimport torch\nimport numpy as np\nimport gc, copy, random, time\nfrom typing import List, Tuple\n\nfrom .model import load_model\nfrom .learn import train_sft, consolidate\nfrom .verify import eval_ppls, check_drift\nfrom .repair import get_lora_state, set_lora_state, reset_lora, repair\nfrom .eval import evaluate, default_scorer\n\n\ndef compute_metrics(R, task_order):\n    T = len(task_order)\n    acc = float(np.mean([R[T-1][j] for j in range(T)]))\n    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]\n    bwt = float(np.mean(bwt_values)) if bwt_values else 0.0\n    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]\n    ff = float(np.mean(ff_values)) if ff_values else 0.0\n    return {"acc": acc, "bwt": bwt, "ff": ff}\n\n\ndef run(model: str,\n        tasks: List[Tuple[str, list, list]],\n        lora_rank: int = 128,\n        lora_alpha: int = 128,\n        lora_targets: list = None,\n        epochs: int = 3,\n        lr: float = 2e-4,\n        batch_size: int = 4,\n        grad_accum: int = 4,\n        ctx_len: int = 512,\n        drift_threshold: float = 1.15,\n        repair_alpha: float = 0.1,\n        max_repair_steps: int = 10,\n        two_stream: bool = False,\n        scorer=None,\n        repair_fn=None,\n        device: str = "cuda",\n        seed: int = 42):\n    """\n    Run Anchor-Verify-Repair on a model + task stream.\n\n    Args:\n        model: HuggingFace model ID\n        tasks: List of (name, train_examples, eval_examples).\n               Each example is (question, answer, gold).\n        lora_rank: LoRA rank. Default 128.\n        lora_targets: LoRA target modules. Auto-detected if None.\n        two_stream: Use hippocampus/neocortex variant. Default False.\n        scorer: Custom scorer(response, gold) -> float. Default: substring match.\n        repair_fn: Custom repair operator with signature\n                   fn(model, snapshot, alpha, device) -> int (num params touched).\n                   Default: avr.repair.repair (linear interpolation toward snapshot).\n                   Pass your own for TIES, TaskArithmetic, etc.\n    """\n    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\n    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)\n\n    task_order = [t[0] for t in tasks]\n    tasks_data = {t[0]: {"train": t[1], "eval": t[2]} for t in tasks}\n    T = len(task_order)\n\n    _do_repair = repair_fn if repair_fn is not None else repair\n\n    print(f"\\n{\'=\'*70}", flush=True)\n    print(f"avr-cl | Model: {model} | Tasks: {task_order}", flush=True)\n    print(f"LoRA r={lora_rank} | Two-stream: {two_stream} | Seed: {seed}", flush=True)\n    print(f"AVR: threshold={drift_threshold}, alpha={repair_alpha}, max_steps={max_repair_steps}", flush=True)\n    print(f"Repair operator: {_do_repair.__name__ if hasattr(_do_repair, \'__name__\') else \'custom\'}", flush=True)\n    print(f"{\'=\'*70}", flush=True)\n\n    model_obj, tokenizer = load_model(model, lora_rank, lora_alpha, lora_targets, device)\n    R = [[0.0]*T for _ in range(T)]\n    best_ppls = {}\n    completed = []\n    total_repairs = 0\n    repair_log = []\n    snapshot = None\n\n    if two_stream:\n        neo_state = get_lora_state(model_obj)\n\n    for ti, task in enumerate(task_order):\n        print(f"\\n{\'=\'*60}\\n  Task {ti+1}/{T}: {task}\\n{\'=\'*60}", flush=True)\n        train_ex = tasks_data[task]["train"]\n\n        # LEARN\n        if two_stream:\n            neo_snapshot = copy.deepcopy(neo_state)\n            print(f"  [twostream] Snapshot taken", flush=True)\n            reset_lora(model_obj)\n            print(f"  [twostream] Hippocampus reset", flush=True)\n            train_sft(model_obj, tokenizer, train_ex, epochs=epochs, lr=lr,\n                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,\n                     device=device, tag="hippo")\n            hippo_state = get_lora_state(model_obj)\n            print(f"  [twostream] Hippocampus trained", flush=True)\n            set_lora_state(model_obj, neo_state, device)\n            neo_state = consolidate(model_obj, tokenizer, hippo_state, neo_state, train_ex,\n                                    epochs=1, lr=lr*0.5, batch_size=batch_size,\n                                    grad_accum=grad_accum, ctx_len=ctx_len, device=device)\n            print(f"  [twostream] Consolidation complete", flush=True)\n            set_lora_state(model_obj, neo_state, device)\n            repair_target = neo_snapshot\n        else:\n            train_sft(model_obj, tokenizer, train_ex, epochs=epochs, lr=lr,\n                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,\n                     device=device, tag="sft")\n            repair_target = snapshot\n\n        # VERIFY\n        post_ppls = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)\n        if task not in best_ppls:\n            best_ppls[task] = post_ppls[task]\n        print(f"  PPLs: " + " | ".join(f"{k}:{v:.2f}" for k,v in post_ppls.items()), flush=True)\n\n        # REPAIR\n        repairs = 0\n        if ti > 0 and repair_target is not None:\n            drifted = check_drift(post_ppls, best_ppls, completed, drift_threshold)\n            if drifted:\n                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)\n                for dk, info in drifted.items():\n                    print(f"    {dk}: {info[\'current\']:.2f}/{info[\'best\']:.2f}={info[\'ratio\']:.2f}x", flush=True)\n                still = drifted\n                for step in range(max_repair_steps):\n                    n = _do_repair(model_obj, repair_target, repair_alpha, device)\n                    repairs += 1\n                    rp = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)\n                    still = check_drift(rp, best_ppls, completed, drift_threshold)\n                    print(f"    [AVR] Repair {step+1}: {n} params, drifted: {list(still.keys()) if still else \'none\'}", flush=True)\n                    if not still:\n                        print(f"  [AVR] Converged at step {step+1}", flush=True)\n                        break\n                if still:\n                    print(f"  [AVR] Max steps reached", flush=True)\n                if two_stream:\n                    neo_state = get_lora_state(model_obj)\n            else:\n                print(f"  [AVR] No drift", flush=True)\n\n        total_repairs += repairs\n        repair_log.append({"task": task, "repairs": repairs})\n\n        # Update best PPLs\n        final_ppls = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)\n        for dk, dp in final_ppls.items():\n            if dk not in best_ppls or dp < best_ppls[dk]:\n                best_ppls[dk] = dp\n\n        # Snapshot\n        if not two_stream:\n            snapshot = get_lora_state(model_obj)\n        completed.append(task)\n\n        # Evaluate (R-matrix)\n        print(f"\\n  Evaluating...", flush=True)\n        for j in range(ti + 1):\n            R[ti][j] = evaluate(model_obj, tokenizer, tasks_data[task_order[j]]["eval"],\n                               task_order[j], scorer=scorer, device=device)\n\n        if torch.cuda.is_available(): torch.cuda.empty_cache()\n        gc.collect()\n\n    metrics = compute_metrics(R, task_order)\n    result = {\n        "acc": metrics["acc"],\n        "bwt": metrics["bwt"],\n        "ff": metrics["ff"],\n        "repairs": total_repairs,\n        "R": R,\n        "repair_log": repair_log,\n        "task_order": task_order,\n    }\n\n    print(f"\\n{\'=\'*70}", flush=True)\n    print(f"RESULTS: ACC={metrics[\'acc\']:.3f}  BWT={metrics[\'bwt\']:+.3f}  FF={metrics[\'ff\']:.3f}  Repairs={total_repairs}", flush=True)\n    print(f"{\'=\'*70}", flush=True)\n\n    del model_obj; gc.collect()\n    if torch.cuda.is_available(): torch.cuda.empty_cache()\n    return result\n',
    'cli.py': '"""\navr.cli — `avr train config.yaml`\n\nConfig format:\n    model: Qwen/Qwen3-1.7B\n    lora_rank: 128\n    tasks:\n      - name: task_a\n        train: data/task_a_train.json\n        eval: data/task_a_eval.json\n      - name: task_b\n        train: data/task_b_train.json\n        eval: data/task_b_eval.json\n\nData format (JSON): list of [question, answer, gold] triples.\n"""\nimport argparse, json, sys\nfrom pathlib import Path\n\n\ndef load_config(path):\n    import yaml\n    with open(path) as f:\n        return yaml.safe_load(f)\n\n\ndef load_task_data(task_cfg):\n    with open(task_cfg["train"]) as f:\n        train = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]\n    with open(task_cfg["eval"]) as f:\n        eval_data = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]\n    return train, eval_data\n\n\ndef main():\n    parser = argparse.ArgumentParser(\n        prog="avr",\n        description="avr-cl: detect when fine-tuning broke old tasks and repair them")\n    sub = parser.add_subparsers(dest="command", required=True)\n\n    train = sub.add_parser("train", help="Run on a task stream")\n    train.add_argument("config", help="Path to YAML config")\n    train.add_argument("--seed", type=int, default=42)\n\n    args = parser.parse_args()\n\n    if args.command == "train":\n        import avr\n        config = load_config(args.config)\n        tasks = []\n        for task_cfg in config["tasks"]:\n            name = task_cfg["name"]\n            train_data, eval_data = load_task_data(task_cfg)\n            tasks.append((name, train_data, eval_data))\n            print(f"  {name}: {len(train_data)} train, {len(eval_data)} eval")\n\n        result = avr.run(\n            model=config["model"],\n            tasks=tasks,\n            lora_rank=config.get("lora_rank", 128),\n            lora_alpha=config.get("lora_alpha", 128),\n            lora_targets=config.get("lora_targets"),\n            epochs=config.get("epochs", 3),\n            lr=config.get("lr", 2e-4),\n            batch_size=config.get("batch_size", 4),\n            grad_accum=config.get("grad_accum", 4),\n            ctx_len=config.get("ctx_len", 512),\n            drift_threshold=config.get("drift_threshold", 1.15),\n            repair_alpha=config.get("repair_alpha", 0.1),\n            max_repair_steps=config.get("max_repair_steps", 10),\n            two_stream=config.get("two_stream", False),\n            seed=args.seed,\n        )\n\n        output = config.get("output", "results.json")\n        with open(output, "w") as f:\n            json.dump(result, f, indent=2, default=str)\n        print(f"\\nResults saved: {output}")\n\n\nif __name__ == "__main__":\n    main()\n',
}


def _inline_avr_package(install_dir="/tmp/avr-inline"):
    """Write the embedded avr package source to disk and add to sys.path.
    No network access needed — the source is inlined in this script."""
    import sys
    from pathlib import Path
    pkg_dir = Path(install_dir) / "avr"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for fn, content in AVR_SOURCES.items():
        (pkg_dir / fn).write_text(content)
    if str(Path(install_dir)) not in sys.path:
        sys.path.insert(0, str(Path(install_dir)))
    # Verify
    import importlib
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("avr")
    if spec is None or spec.origin is None:
        raise RuntimeError("avr package not findable after inline write")
    print(f"[bootstrap] avr package inlined to {install_dir}", flush=True)
    return "inlined"

def _modelscope_download_direct(model_id, cache_dir):
    """Download model files directly from ModelScope HTTP API using urllib.
    No modelscope pip package needed — just stdlib urllib.
    ModelScope API: https://www.modelscope.cn/api/v1/models/{model_id}/repo?Revision=master&FilePath={path}
    """
    import urllib.request, json
    cache_dir = Path(cache_dir)
    # ModelScope stores under models/{org}--{name}/snapshots/master/
    org, name = model_id.split("/", 1)
    model_dir = cache_dir / "models" / f"{org}--{name}" / "snapshots" / "master"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: get file list
    list_url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo/files?Revision=master"
    print(f"  [ms-direct] Fetching file list...", flush=True)
    req = urllib.request.Request(list_url, headers={"User-Agent": "avr-cl/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    files = resp.get("Data", {}).get("Files", [])
    if not files:
        raise RuntimeError(f"No files found for {model_id} on ModelScope")

    # Step 2: download each file (skip hidden files like .gitattributes)
    for f in files:
        path = f["Path"]
        size = f.get("Size", 0)
        if path.startswith(".") or size == 0:
            continue
        dest = model_dir / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == size:
            print(f"  [ms-direct] {path} (cached, {size:,} bytes)", flush=True)
            continue
        dl_url = f"https://www.modelscope.cn/api/v1/models/{model_id}/repo?Revision=master&FilePath={path}"
        print(f"  [ms-direct] Downloading {path} ({size:,} bytes)...", flush=True)
        req = urllib.request.Request(dl_url, headers={"User-Agent": "avr-cl/0.1"})
        with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as out:
            shutil.copyfileobj(r, out)
        print(f"  [ms-direct]   OK ({dest.stat().st_size:,} bytes)", flush=True)

    print(f"  [ms-direct] All files downloaded to {model_dir}", flush=True)
    return str(model_dir)



def download_model(model_id, cache_dir, prefer="modelscope"):
    """
    Download a model snapshot. Returns the local path.

    Tries in order:
      1. ModelScope snapshot_download  (fast in China; sometimes flaky on Kaggle)
      2. HuggingFace via hf-mirror.com with xet disabled  (HF_HUB_DISABLE_XET=1)
      3. Direct HuggingFace with xet disabled  (works from US/West Kaggle IPs)

    Args:
        model_id: e.g. "Qwen/Qwen3-1.7B" or "liquidai/LFM2.5-1.2B-Instruct"
        cache_dir: where to cache. Will be created if missing.
        prefer: 'modelscope' (try MS first) or 'hf' (skip MS).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    errors = []

    # --- Source 1: Direct HTTP from ModelScope API (no pip package) ---
    if prefer != "hf":
        try:
            path = _modelscope_download_direct(model_id, cache_dir)
            print(f"[download] ModelScope direct HTTP OK: {path}", flush=True)
            return path
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            errors.append(f"modelscope-direct: {msg}")
            print(f"[download] ModelScope direct HTTP failed: {msg}", flush=True)

    # --- Source 2: ModelScope (optional — may not be installed) ---
    if prefer != "hf":
        try:
            from modelscope import snapshot_download as ms_download
            path = ms_download(model_id, cache_dir=str(cache_dir))
            print(f"[download] ModelScope OK: {path}", flush=True)
            return path
        except ImportError:
            errors.append("modelscope: not installed (optional, skipped)")
            print(f"[download] ModelScope: not installed, skipping to HF", flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {str(e)[:200]}"
            errors.append(f"modelscope: {msg}")
            print(f"[download] ModelScope failed: {msg}", flush=True)

    # --- Source 2: HuggingFace mirror (hf-mirror.com) with xet disabled ---
    # HF_HUB_DISABLE_XET=1 is already set at module load. Reassert + endpoint.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download as hf_download
        path = hf_download(model_id, cache_dir=str(cache_dir))
        print(f"[download] HF mirror (hf-mirror.com) OK: {path}", flush=True)
        return path
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        errors.append(f"hf-mirror: {msg}")
        print(f"[download] HF mirror failed: {msg}", flush=True)

    # --- Source 3: Direct HuggingFace (no mirror) with xet disabled ---
    os.environ.pop("HF_ENDPOINT", None)  # let it resolve to default
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import snapshot_download as hf_download
        path = hf_download(model_id, cache_dir=str(cache_dir))
        print(f"[download] Direct HF OK: {path}", flush=True)
        return path
    except Exception as e:
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        errors.append(f"hf-direct: {msg}")
        print(f"[download] Direct HF failed: {msg}", flush=True)

    raise RuntimeError(
        f"All model download strategies failed for {model_id}:\n" +
        "\n".join(f"  - {e}" for e in errors))


# ---------------------------------------------------------------------------
# 4. transformers/torch 2.6 workaround
# ---------------------------------------------------------------------------
def patch_transformers_torch26():
    """
    Kaggle ships torch>=2.6 but some transformers versions gate features
    behind is_torch_greater_or_equal_than_2_6 in a way that breaks on T4.
    Force the check to False so transformers uses the 2.5 code path.
    """
    try:
        import transformers.utils.import_utils as _iu
        _iu._is_torch_greater_or_equal_than_2_6 = False
        _iu.is_torch_greater_or_equal_than_2_6 = lambda: False
        print("[bootstrap] Patched transformers torch-2.6 check -> False", flush=True)
    except (AttributeError, ImportError):
        pass

install_deps()                       # peft, accelerate, modelscope, huggingface_hub, ...
_inline_avr_package()                               # inlined avr source, no network needed
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

MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ============================================================================
# Data loaders (same as exp1 — GitHub raw, no HF)
# ============================================================================

import re, random, math, gc, json, time, copy, shutil
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ============================================================================
# TRACE DATA — download from Google Drive (works on Kaggle, different DNS than GitHub)
# ============================================================================
TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
TRACE_TASKS = ["C-STANCE", "FOMC", "MeetingBank", "Py150",
               "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"]

def load_trace():
    """Download TRACE 0.5K from Google Drive and load all 8 tasks.
    Truncates prompts to ~1500 chars (~500 tokens) to keep training feasible —
    MeetingBank and Py150 have very long examples that would take 24+ hours each."""
    import gdown, zipfile
    trace_dir = OUTPUT_DIR / "trace_data"
    if not (trace_dir.exists() and any(trace_dir.iterdir())):
        print("  Downloading TRACE from Google Drive...", flush=True)
        zip_path = OUTPUT_DIR / "trace_benchmark.zip"
        gdown.download(id=TRACE_GDRIVE_ID, output=str(zip_path), quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(OUTPUT_DIR)
        # Find the 500 variant
        for d in OUTPUT_DIR.rglob("LLM-CL-Benchmark_500"):
            if d.is_dir():
                trace_dir = d; break
    else:
        # Already downloaded, find it
        for d in OUTPUT_DIR.rglob("LLM-CL-Benchmark_500"):
            if d.is_dir():
                trace_dir = d; break

    MAX_PROMPT_CHARS = 1500  # ~500 tokens, matches ctx_len=512
    rng = random.Random(SEED)
    tasks = []
    for task_name in TRACE_TASKS:
        task_dir = trace_dir / task_name
        with open(task_dir / "train.json") as f: train_data = json.load(f)
        with open(task_dir / "test.json") as f: test_data = json.load(f)
        # Truncate long prompts — critical for MeetingBank and Py150
        train_pairs = []
        for ex in train_data:
            prompt = ex["prompt"][:MAX_PROMPT_CHARS] if len(ex["prompt"]) > MAX_PROMPT_CHARS else ex["prompt"]
            train_pairs.append((prompt, ex["answer"], ex["answer"]))
        test_pairs = []
        for ex in test_data:
            prompt = ex["prompt"][:MAX_PROMPT_CHARS] if len(ex["prompt"]) > MAX_PROMPT_CHARS else ex["prompt"]
            test_pairs.append((prompt, ex["answer"], ex["answer"]))
        rng.shuffle(train_pairs); rng.shuffle(test_pairs)
        tasks.append((task_name, train_pairs, test_pairs))
        # Show token estimate
        total_chars = sum(len(p) for p, _, _ in train_pairs)
        print(f"  {task_name}: {len(train_pairs)} train, {len(test_pairs)} eval, "
              f"~{total_chars//4:,} tokens (truncated to {MAX_PROMPT_CHARS} chars)", flush=True)
    return tasks

# EXPERIMENT CONFIG
# ============================================================================
SEED = 42
MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRACE_TASKS = ['C-STANCE', 'FOMC', 'MeetingBank', 'Py150', 'ScienceQA', 'NumGLUE-cm', 'NumGLUE-ds', '20Minuten']

# ============================================================================
# MAIN
# ============================================================================
print("="*70, flush=True)
print("EXP 3: 8-Task TRACE, AVR-only (no Two-Stream)", flush=True)
print("Naive vs AVR on 8 diverse tasks", flush=True)
print("="*70, flush=True)

print("\nLoading TRACE 0.5K (embedded)...", flush=True)
tasks = load_trace()

print(f"\nDownloading {MODEL_ID} (ModelScope direct HTTP)...", flush=True)
MODEL_PATH = download_model(MODEL_ID, cache_dir=OUTPUT_DIR / "model_cache")
print(f"  Cached: {MODEL_PATH}", flush=True)

# Condition A: Naive
print(f"\n{'#'*60}\n# Condition A: Naive (8 tasks)\n{'#'*60}", flush=True)
try:
    result_naive = avr.run(
        model=MODEL_PATH,
        tasks=tasks,
        lora_rank=32,
        lora_alpha=32,
        lora_targets=LORA_TARGETS,
        epochs=3,
        lr=2e-4,
        batch_size=4,
        grad_accum=4,
        ctx_len=512,
        drift_threshold=999.0,
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
        model=MODEL_PATH,
        tasks=tasks,
        lora_rank=32,
        lora_alpha=32,
        lora_targets=LORA_TARGETS,
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
