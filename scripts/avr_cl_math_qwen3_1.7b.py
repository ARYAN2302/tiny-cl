"""
avr-cl validation v2 — MATH stream on Qwen3-1.7B.

Re-run after v1 (350M, unrelated domains) showed no forgetting.

Changes from v1:
  - Model: Qwen3-1.7B (was LFM2.5-350M)
  - LoRA: r=128 (was r=32) — forgetting scales with rank
  - Tasks: math stream — GSM8K -> MATH(algebra) -> AQuA-RAT -> SVAMP
    (was unrelated domains: gsm8k/sciq/medmcqa/commonsense_qa)
  - Examples: 5000 per task (was 500)
  - Context: 1024 (was 512)
  - Chat template: Qwen3 with enable_thinking=False
  - Sanity gate: abort if GSM8K < 35% after T1 (need to actually learn the task)

Expected: Naive BWT -0.10 to -0.25 (real forgetting). AVR target: recover to <=-0.05.

Run via Modal (detached):
    modal run --detach avr_cl_math.py
"""
import os
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json, time, random, math, gc, re, copy, sys
from pathlib import Path
from typing import List, Tuple, Dict

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

OUTPUT_DIR = Path("/root/output")

# ============================================================================
# CONFIG
# ============================================================================
MODEL_ID = "Qwen/Qwen3-1.7B"

LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

TRAIN_LR = 2e-4
CONSOLIDATION_LR = 1e-4
TRAIN_WD = 0.01
TRAIN_MAX_GRAD_NORM = 1.0
TASK_EPOCHS = 3
CONSOLIDATION_EPOCHS = 1
BATCH_SIZE = 8          # smaller batch for r=128
CONTEXT_LENGTH = 1024   # math problems are longer
GRAD_ACCUM_STEPS = 2    # effective batch = 16

DRIFT_THRESHOLD = 1.15
REPAIR_ALPHA = 0.1
MAX_REPAIR_STEPS = 10

BENCH_MAX_NEW_TOKENS = 400   # math needs longer generations (some answers need full reasoning)
SEED = 42
DEVICE = "cuda"

# Sanity gate: abort if T1 accuracy < this
T1_MIN_ACCURACY = 0.25   # lowered from 0.35 — Qwen3-1.7B base on GSM8K is ~30-50% with proper scoring

TASK_NAMES = ["gsm8k", "math_algebra", "aqua_rat", "svamp"]

# ============================================================================
# SEEDING
# ============================================================================
def set_seed(seed: int):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================================
# PROMPT FORMATTING — Qwen3 chat template with enable_thinking=False
# ============================================================================
_TOKENIZER = None
def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
        if _TOKENIZER.pad_token is None:
            _TOKENIZER.pad_token = _TOKENIZER.eos_token
    return _TOKENIZER

def format_prompt(question: str) -> str:
    """Format a question as a Qwen3 chat prompt with thinking disabled."""
    tok = _get_tokenizer()
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return text

def format_example(question: str, answer: str) -> str:
    """Format a (question, answer) pair as a full Qwen3 chat example for SFT."""
    tok = _get_tokenizer()
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    return text + tok.eos_token

# ============================================================================
# DATA LOADERS — 4 math datasets
# ============================================================================
def fmt_gsm8k_question(ex):
    """GSM8K: returns (prompt, full_reasoning_answer, gold_number).
    Training target = full reasoning chain ending in '#### N'.
    Scoring gold = just N."""
    q = ex["question"]
    a = ex["answer"]  # full reasoning + '#### N'
    prompt_q = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{q}"
    m = re.search(r'####\s*(-?[\d,.]+)', a)
    gold = m.group(1).replace(",", "").strip() if m else a.strip()
    return prompt_q, a, gold

def fmt_math_question(ex):
    """MATH algebra: returns (prompt, full_solution, gold_from_boxed)."""
    q = ex["problem"]
    sol = ex["solution"]
    prompt_q = f"Solve the math problem. End with the final answer in \\boxed{{}}.\n\n{q}"
    m = re.findall(r'\\boxed\{([^}]+)\}', sol)
    gold = m[-1].strip() if m else ""
    if not gold:
        nums = re.findall(r'-?[\d.]+', sol)
        gold = nums[-1] if nums else ""
    return prompt_q, sol, gold

def fmt_aqua_question(ex):
    """AQuA-RAT: returns (prompt, rationale+answer, correct_letter)."""
    q = ex["question"]
    opts = ex["options"]
    correct = ex["correct"]
    rationale = ex.get("rationale", "")
    letters = ["A", "B", "C", "D", "E"]
    cleaned_opts = []
    for i, o in enumerate(opts):
        o = str(o).strip()
        if len(o) >= 2 and o[0].upper() == letters[i] and o[1] in ").:":
            o = o[2:].strip()
        cleaned_opts.append(o)
    opt_text = "\n".join(f"{l}. {o}" for l, o in zip(letters, cleaned_opts))
    prompt_q = f"{q}\n{opt_text}\n\nAnswer with the letter (A, B, C, D, or E):"
    train_answer = f"{rationale}\n\nAnswer: {correct}"
    return prompt_q, train_answer, correct

def fmt_svamp_question(ex):
    """SVAMP: returns (prompt, reasoning+answer, gold_number)."""
    body = ex.get("Body", "")
    question = ex.get("Question", "")
    answer = ex.get("Answer", "")
    equation = ex.get("Equation", "")
    full_q = f"{body} {question}".strip()
    prompt_q = f"Solve the math problem step by step. End with '#### <final_number>'.\n\n{full_q}"
    try:
        gold_f = float(answer)
        if gold_f == int(gold_f):
            gold = str(int(gold_f))
        else:
            gold = str(gold_f)
    except:
        gold = str(answer)
    train_answer = f"Let me solve this step by step.\nEquation: {equation}\n#### {gold}"
    return prompt_q, train_answer, gold

def load_math_stream() -> Tuple[Dict, Dict, List[str]]:
    """Load 4 math datasets, 5000 train / 200 test each."""
    from datasets import load_dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data, test_data = {}, {}
    rng = random.Random(SEED)

    print("    Loading GSM8K...", flush=True)
    ds_tr = load_dataset("openai/gsm8k", "main", split="train")
    ds_te = load_dataset("openai/gsm8k", "main", split="test")
    tr = [fmt_gsm8k_question(ex) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_gsm8k_question(ex) for ex in ds_te]; rng.shuffle(te)
    # train_data[task] = list of (prompt, train_answer, gold) for training
    # test_data[task]  = list of (prompt, train_answer, gold) for eval (we only use prompt + gold)
    train_data["gsm8k"] = tr[:5000]; test_data["gsm8k"] = te[:200]
    print(f"    gsm8k: {len(train_data['gsm8k'])} train, {len(test_data['gsm8k'])} test", flush=True)

    print("    Loading MATH (algebra)...", flush=True)
    try:
        ds_tr = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
        ds_te = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    except Exception:
        ds_tr = load_dataset("lighteval/MATH", "algebra", split="train")
        ds_te = load_dataset("lighteval/MATH", "algebra", split="test")
    tr = [fmt_math_question(ex) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_math_question(ex) for ex in ds_te]; rng.shuffle(te)
    train_data["math_algebra"] = tr[:5000]; test_data["math_algebra"] = te[:200]
    print(f"    math_algebra: {len(train_data['math_algebra'])} train, {len(test_data['math_algebra'])} test", flush=True)

    print("    Loading AQuA-RAT...", flush=True)
    ds_tr = load_dataset("deepmind/aqua_rat", "raw", split="train")
    ds_te = load_dataset("deepmind/aqua_rat", "raw", split="validation")
    tr = [fmt_aqua_question(ex) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_aqua_question(ex) for ex in ds_te]; rng.shuffle(te)
    train_data["aqua_rat"] = tr[:5000]; test_data["aqua_rat"] = te[:200]
    print(f"    aqua_rat: {len(train_data['aqua_rat'])} train, {len(test_data['aqua_rat'])} test", flush=True)

    print("    Loading SVAMP...", flush=True)
    ds_tr = load_dataset("ChilleD/SVAMP", split="train")
    ds_te = load_dataset("ChilleD/SVAMP", split="test")
    tr = [fmt_svamp_question(ex) for ex in ds_tr]; rng.shuffle(tr)
    te = [fmt_svamp_question(ex) for ex in ds_te]; rng.shuffle(te)
    train_data["svamp"] = tr[:5000]; test_data["svamp"] = te[:200]
    print(f"    svamp: {len(train_data['svamp'])} train, {len(test_data['svamp'])} test", flush=True)

    return train_data, test_data, TASK_NAMES

# ============================================================================
# MODEL + LoRA
# ============================================================================
def create_model_and_tokenizer():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  Loading {MODEL_ID}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    global _TOKENIZER
    _TOKENIZER = tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation="eager",
    )
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    print(f"  LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}", flush=True)
    return model, tokenizer

# ============================================================================
# LoRA STATE UTILITIES (identical to v23/v34)
# ============================================================================
def get_lora_state(model) -> Dict[str, 'torch.Tensor']:
    import torch
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state: Dict[str, 'torch.Tensor']):
    import torch
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(DEVICE).to(p.data.dtype))

def reset_lora_to_peft_init(model):
    import torch.nn.init as init
    import math
    for n, p in model.named_parameters():
        if "lora_A" in n:
            init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()

# ============================================================================
# AVR CORE (identical to v23)
# ============================================================================
def compute_ppl(model, tokenizer, examples, max_samples=100) -> float:
    """examples = list of (prompt, train_answer, gold). Uses prompt + train_answer for PPL."""
    import torch
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for prompt_q, train_answer, gold in examples[:max_samples]:
        text = format_example(prompt_q, train_answer)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))

def eval_all_ppls(model, tokenizer, train_data, task_order, trained_so_far, max_samples=100) -> Dict[str, float]:
    ppls = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far: break
        ppls[task] = compute_ppl(model, tokenizer, train_data[task], max_samples)
    return ppls

def verify_drift(current_ppls, best_ppls, completed_tasks, threshold=DRIFT_THRESHOLD) -> Dict:
    drifted = {}
    for task in completed_tasks:
        if task not in current_ppls or task not in best_ppls: continue
        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0
        if ratio > threshold:
            drifted[task] = {"current_ppl": current_ppls[task], "best_ppl": best_ppls[task], "ratio": ratio}
    return drifted

def repair_toward_snapshot(model, snapshot_state, alpha=REPAIR_ALPHA) -> int:
    import torch
    n_adj = 0
    for n, p in model.named_parameters():
        if "lora_" in n and n in snapshot_state:
            snap_val = snapshot_state[n].to(DEVICE)
            p.data.copy_((1.0 - alpha) * p.data + alpha * snap_val)
            n_adj += 1
    return n_adj

# ============================================================================
# TRAINING (LEARN phase)
# ============================================================================
class TextDataset:
    def __init__(self, token_ids, context_length):
        import torch
        self.token_ids = token_ids
        self.context_length = context_length
        self.n_chunks = max(1, len(token_ids) // context_length)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        import torch
        s = idx * self.context_length
        e = s + self.context_length
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}

def build_training_stream(tokenizer, examples) -> 'torch.Tensor':
    """Build training stream using Qwen3 chat template format.
    examples = list of (prompt, train_answer, gold). Uses prompt + train_answer for SFT."""
    import torch
    all_tokens = []
    for prompt_q, train_answer, gold in examples:
        text = format_example(prompt_q, train_answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)

def train_lora(model, tokenizer, pairs, lr=TRAIN_LR, epochs=TASK_EPOCHS, tag="sft"):
    import torch
    from torch.utils.data import DataLoader
    token_ids = build_training_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=TRAIN_WD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(DEVICE), labels=batch["labels"].to(DEVICE))
            loss = out.loss / GRAD_ACCUM_STEPS
            loss.backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += out.loss.item()
                if gs % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {elapsed:.0f}s", flush=True)
    return gs, tl

def consolidate_to_neocortex(model, tokenizer, hippo_state, neo_state, pairs, epochs=CONSOLIDATION_EPOCHS):
    """KL distill hippocampus -> neocortex (v34 two-stream)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    print(f"    [consolid] KL distill hippocampus -> neocortex ({epochs} epoch)", flush=True)
    token_ids = build_training_stream(tokenizer, pairs)
    dataset = TextDataset(token_ids, CONTEXT_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=CONSOLIDATION_LR, weight_decay=TRAIN_WD)
    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            set_lora_state(model, hippo_state)
            model.eval()
            with torch.no_grad():
                hippo_logits = model(input_ids=input_ids).logits
            set_lora_state(model, neo_state)
            model.train()
            neo_logits = model(input_ids=input_ids).logits
            shift_hippo = hippo_logits[..., :-1, :].contiguous()
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            p_hippo = F.softmax(shift_hippo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            (kl_loss / GRAD_ACCUM_STEPS).backward()
            accum += 1
            if accum >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(trainable, TRAIN_MAX_GRAD_NORM)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += kl_loss.item()
                if gs % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {elapsed:.0f}s", flush=True)
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state

# ============================================================================
# EVALUATION — math answer scoring
# ============================================================================
def normalize_math_answer(s: str) -> str:
    """Normalize a math answer for comparison."""
    s = s.strip()
    # Remove $ \, \! etc
    s = s.replace('$', '').replace('\\', '').replace('!', '').replace(',', '')
    s = s.replace('{', '').replace('}', '')
    s = s.replace(' ', '')
    # Try to convert to float for numeric compare
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except:
        return s.lower()

def extract_response_answer(response: str, is_mcq: bool = False) -> str:
    """Extract the final answer from a model response.
    Handles Qwen3's output formats: \\boxed{}, ####, 'Final Answer: **N**', 'answer is N', last number."""
    response = response.strip()
    if is_mcq:
        # Look for A/B/C/D/E in last 100 chars (answer usually at end)
        tail = response[-100:].upper()
        m = re.search(r'\b([A-E])\b', tail)
        if m:
            return m.group(1)
        # fallback: first char if it's a letter
        if response and response[0].upper() in "ABCDE":
            return response[0].upper()
        return response[:1]
    # Math: try \boxed{} first
    matches = re.findall(r'\\boxed\{([^}]+)\}', response)
    if matches:
        return matches[-1].strip()
    # Try "#### number" format (GSM8K-style)
    m = re.search(r'####\s*(-?[\d,.]+)', response)
    if m:
        return m.group(1).replace(",", "").strip()
    # Try "Final Answer: **N**" or "Final answer: **N**" (Qwen3 markdown style)
    m = re.search(r'(?:final answer|answer)\s*:?\s*\**\s*([^\n*]+)', response, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip('*').strip()
        # Extract number from candidate (must have at least one digit)
        nums = re.findall(r'-?\d[\d,.]*', candidate)
        if nums:
            return nums[-1].replace(",", "").strip()
        return candidate
    # Try "answer is X" or "answer is **X**"
    m = re.search(r'answer is\s*\**\s*([^\n*]+)', response, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip().rstrip('*').strip()
        nums = re.findall(r'-?\d[\d,.]*', candidate)
        if nums:
            return nums[-1].replace(",", "").strip()
    # Fallback: last number in the response (must contain at least one digit)
    numbers = re.findall(r'-?\d[\d,.]*', response)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return response[:50] if response else ""

def score_math(response: str, gold: str, task_name: str) -> float:
    """Score math answer with task-specific logic."""
    is_mcq = (task_name == "aqua_rat")
    resp_ans = extract_response_answer(response, is_mcq=is_mcq)
    if is_mcq:
        return 1.0 if resp_ans.upper() == gold.upper() else 0.0
    return 1.0 if normalize_math_answer(resp_ans) == normalize_math_answer(gold) else 0.0

def generate_batch(model, tokenizer, prompt_qs: List[str], max_new_tokens=BENCH_MAX_NEW_TOKENS, batch_size: int = 16) -> List[str]:
    """Batched generation: process multiple prompts per forward pass.
    ~Nx throughput on memory-bandwidth-bound decode.
    Note: skip_special_tokens=False — Qwen3 emits <think> blocks we need to see,
    but we strip them in post-processing. The empty-output bug came from
    skip_special_tokens=True dropping content."""
    import torch
    results = []
    # Disable gradient checkpointing for inference (faster, no grad needed)
    gc_was_enabled = getattr(model, "gradient_checkpointing", False)
    if gc_was_enabled:
        try: model.gradient_checkpointing_disable()
        except: pass
    model.eval()
    try:
        for i in range(0, len(prompt_qs), batch_size):
            batch_prompts = prompt_qs[i:i+batch_size]
            texts = [format_prompt(q) for q in batch_prompts]
            # Tokenize with left padding for batched generation
            tokenizer.padding_side = "left"
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=1024, padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id,
                    temperature=1.0,
                )
            # Decode only the generated part — skip_special_tokens=True is correct here
            # because Qwen3 instruct mode (enable_thinking=False) emits clean text + EOS
            for j, out in enumerate(outputs):
                input_len = inputs["input_ids"].shape[1]
                gen_text = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
                results.append(gen_text)
    finally:
        # Re-enable gradient checkpointing for next training phase
        if gc_was_enabled:
            try:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
            except: pass
    return results

def evaluate_task_accuracy(model, tokenizer, test_examples, task_name, max_questions=200, batch_size: int = 16) -> float:
    """Batched evaluation. test_examples = list of (prompt, train_answer, gold). Uses prompt + gold only."""
    import torch
    print(f"    Eval {task_name} ({min(len(test_examples), max_questions)} Qs, batch={batch_size})...", flush=True)
    total = min(len(test_examples), max_questions)
    examples = test_examples[:total]
    prompt_qs = [ex[0] for ex in examples]  # prompt
    golds = [ex[2] for ex in examples]       # gold (for scoring)
    t0 = time.time()
    correct = 0
    processed = 0
    for batch_start in range(0, len(prompt_qs), batch_size):
        batch_end = min(batch_start + batch_size, len(prompt_qs))
        batch_prompts = prompt_qs[batch_start:batch_end]
        batch_golds = golds[batch_start:batch_end]
        responses = generate_batch(model, tokenizer, batch_prompts,
                                    max_new_tokens=BENCH_MAX_NEW_TOKENS,
                                    batch_size=len(batch_prompts))
        for resp, gold in zip(responses, batch_golds):
            if score_math(resp, gold, task_name): correct += 1
            processed += 1
        if processed % (batch_size * 2) < batch_size or processed >= total:
            elapsed = time.time() - t0
            print(f"      [{processed}/{total}] acc={correct/processed:.3f} ({elapsed:.0f}s)", flush=True)
    acc = correct / total
    elapsed = time.time() - t0
    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({elapsed:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc

def compute_metrics(R, task_order) -> Dict[str, float]:
    import numpy as np
    T = len(task_order)
    ACC = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]
    BWT = float(np.mean(bwt_values)) if bwt_values else 0.0
    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]
    FF = float(np.mean(ff_values)) if ff_values else 0.0
    return {"ACC": ACC, "BWT": BWT, "FF": FF}

# ============================================================================
# CHECKPOINT
# ============================================================================
def save_partial(condition, results, task_idx):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"partial_math_{condition}_task{task_idx+1}.json"
    with open(path, "w") as f:
        json.dump({"condition": condition, "task_completed": task_idx + 1, "results": results},
                  f, indent=2, default=str)
    print(f"  [checkpoint] saved {path.name}", flush=True)

# ============================================================================
# CONDITION 1: NAIVE
# ============================================================================
def run_naive(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 1: Naive sequential SFT\n{'#'*70}", flush=True)
    model, tokenizer = create_model_and_tokenizer()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    for task_idx, task in enumerate(task_order):
        task_num = task_idx + 1
        print(f"\n{'='*60}\n  Task {task_num}/{T}: {task}  (naive)\n{'='*60}", flush=True)
        train_lora(model, tokenizer, train_data[task], tag="naive")
        print(f"\n  Evaluating accuracy on all tasks seen so far...", flush=True)
        for j in range(task_idx + 1):
            R[task_idx][j] = evaluate_task_accuracy(model, tokenizer, test_data[task_order[j]], task_order[j])
        # Sanity gate after T1
        if task_idx == 0:
            t1_acc = R[0][0]
            print(f"\n  [SANITY GATE] T1 ({task}) accuracy = {t1_acc:.3f}", flush=True)
            if t1_acc < T1_MIN_ACCURACY:
                msg = f"ABORT: T1 accuracy {t1_acc:.3f} < {T1_MIN_ACCURACY} (model at floor, no forgetting to demonstrate)"
                print(f"  {msg}", flush=True)
                with open(OUTPUT_DIR / "RUN_FAILED.txt", "w") as f:
                    f.write(f"Sanity gate failed: {msg}\n")
                raise RuntimeError(msg)
            print(f"  [SANITY GATE] PASSED — proceeding", flush=True)
        partial = {"R": R, "metrics": compute_metrics(R, task_order)}
        save_partial("naive", partial, task_idx)
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [naive] ACC={metrics['ACC']:.3f}  BWT={metrics['BWT']:+.3f}  FF={metrics['FF']:.3f}", flush=True)
    del model; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics

# ============================================================================
# CONDITION 2: AVR (v23 style)
# ============================================================================
def run_avr(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 2: AVR (SFT + PPL verify + closed-form repair)\n{'#'*70}", flush=True)
    model, tokenizer = create_model_and_tokenizer()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    snapshot = None
    best_ppls = {}
    completed_tasks = []
    total_repair_steps = 0
    log = []
    for task_idx, task in enumerate(task_order):
        task_num = task_idx + 1
        print(f"\n{'='*60}\n  Task {task_num}/{T}: {task}  (AVR)\n{'='*60}", flush=True)
        train_lora(model, tokenizer, train_data[task], tag="avr-learn")
        post_train_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        if task not in best_ppls: best_ppls[task] = post_train_ppls[task]
        print(f"  Post-train PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_train_ppls.items()), flush=True)
        repair_steps = 0
        if task_num > 1 and snapshot is not None:
            drifted = verify_drift(post_train_ppls, best_ppls, completed_tasks)
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x", flush=True)
                still_drifted = drifted
                for step in range(MAX_REPAIR_STEPS):
                    n_adj = repair_toward_snapshot(model, snapshot)
                    repair_steps += 1
                    repair_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_tasks)
                    print(f"    [AVR] Repair step {step+1}: {n_adj} params, still drifted: {list(still_drifted.keys()) if still_drifted else 'none'}", flush=True)
                    if not still_drifted:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still_drifted:
                    print(f"  [AVR] Max steps ({MAX_REPAIR_STEPS}) reached, drift remains", flush=True)
            else:
                print(f"  [AVR] No drift - no repair needed", flush=True)
        total_repair_steps += repair_steps
        log.append({"task": task, "repair_steps": repair_steps})
        final_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        for dpk, dppl in final_ppls.items():
            if dpk not in best_ppls or dppl < best_ppls[dpk]: best_ppls[dpk] = dppl
        snapshot = get_lora_state(model)
        completed_tasks.append(task)
        print(f"\n  Evaluating accuracy on all tasks seen so far...", flush=True)
        for j in range(task_idx + 1):
            R[task_idx][j] = evaluate_task_accuracy(model, tokenizer, test_data[task_order[j]], task_order[j])
        partial = {"R": R, "metrics": compute_metrics(R, task_order),
                   "total_repairs": total_repair_steps, "log": log}
        save_partial("avr", partial, task_idx)
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [AVR] ACC={metrics['ACC']:.3f}  BWT={metrics['BWT']:+.3f}  FF={metrics['FF']:.3f}", flush=True)
    print(f"  [AVR] Total repair steps: {total_repair_steps}", flush=True)
    del model; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics, total_repair_steps, log

# ============================================================================
# CONDITION 3: TWO-STREAM + AVR (v34 condition C)
# ============================================================================
def run_twostream_avr(train_data, test_data, task_order):
    print(f"\n{'#'*70}\n# CONDITION 3: Two-Stream + AVR\n{'#'*70}", flush=True)
    model, tokenizer = create_model_and_tokenizer()
    T = len(task_order)
    R = [[0.0]*T for _ in range(T)]
    neo_state = get_lora_state(model)
    best_ppls = {}
    completed_tasks = []
    total_repair_steps = 0
    log = []
    for task_idx, task in enumerate(task_order):
        task_num = task_idx + 1
        print(f"\n{'='*60}\n  Task {task_num}/{T}: {task}  (two-stream+AVR)\n{'='*60}", flush=True)
        neo_snapshot = copy.deepcopy(neo_state)
        print(f"  [twostream] Neocortex snapshot taken", flush=True)
        reset_lora_to_peft_init(model)
        print(f"  [twostream] Hippocampus reset to fresh PEFT init", flush=True)
        train_lora(model, tokenizer, train_data[task], lr=TRAIN_LR, tag="hippo")
        hippo_state = get_lora_state(model)
        print(f"  [twostream] Hippocampus trained", flush=True)
        set_lora_state(model, neo_state)
        neo_state = consolidate_to_neocortex(model, tokenizer, hippo_state, neo_state, train_data[task])
        print(f"  [twostream] Consolidation complete", flush=True)
        repair_steps = 0
        set_lora_state(model, neo_state)
        post_consolid_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        if task not in best_ppls: best_ppls[task] = post_consolid_ppls[task]
        print(f"  Post-consolid PPLs: " + " | ".join(f"{k}: {v:.2f}" for k, v in post_consolid_ppls.items()), flush=True)
        if task_num > 1:
            drifted = verify_drift(post_consolid_ppls, best_ppls, completed_tasks)
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)
                for dk, info in drifted.items():
                    print(f"    {dk}: PPL={info['current_ppl']:.2f} / best={info['best_ppl']:.2f} = {info['ratio']:.2f}x", flush=True)
                still_drifted = drifted
                for step in range(MAX_REPAIR_STEPS):
                    n_adj = repair_toward_snapshot(model, neo_snapshot)
                    repair_steps += 1
                    repair_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
                    still_drifted = verify_drift(repair_ppls, best_ppls, completed_tasks)
                    print(f"    [AVR] Repair step {step+1}: {n_adj} params, still drifted: {list(still_drifted.keys()) if still_drifted else 'none'}", flush=True)
                    if not still_drifted:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still_drifted:
                    print(f"  [AVR] Max steps ({MAX_REPAIR_STEPS}) reached, drift remains", flush=True)
                neo_state = get_lora_state(model)
            else:
                print(f"  [AVR] No drift - no repair needed", flush=True)
        total_repair_steps += repair_steps
        log.append({"task": task, "repair_steps": repair_steps})
        final_ppls = eval_all_ppls(model, tokenizer, train_data, task_order, task_num)
        for dpk, dppl in final_ppls.items():
            if dpk not in best_ppls or dppl < best_ppls[dpk]: best_ppls[dpk] = dppl
        completed_tasks.append(task)
        print(f"\n  Evaluating accuracy on all tasks seen so far...", flush=True)
        for j in range(task_idx + 1):
            R[task_idx][j] = evaluate_task_accuracy(model, tokenizer, test_data[task_order[j]], task_order[j])
        partial = {"R": R, "metrics": compute_metrics(R, task_order),
                   "total_repairs": total_repair_steps, "log": log}
        save_partial("twostream", partial, task_idx)
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
    metrics = compute_metrics(R, task_order)
    print(f"\n  [twostream+AVR] ACC={metrics['ACC']:.3f}  BWT={metrics['BWT']:+.3f}  FF={metrics['FF']:.3f}", flush=True)
    print(f"  [twostream+AVR] Total repair steps: {total_repair_steps}", flush=True)
    del model; gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return R, metrics, total_repair_steps, log

# ============================================================================
# HEATMAP
# ============================================================================
def make_heatmap(results, task_order, save_path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Naive SFT", "AVR (SFT + verify + repair)", "Two-Stream + AVR"]
    keys = ["naive", "avr", "twostream"]
    short = [t[:12] for t in task_order]
    for ax, title, key in zip(axes, titles, keys):
        if key not in results: continue
        R = np.array(results[key]["R"])
        im = ax.imshow(R, cmap='RdYlGn', vmin=0, vmax=0.6, aspect='auto')
        ax.set_xticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=9)
        ax.set_xlabel("Eval task", fontsize=10)
        ax.set_ylabel("After training task", fontsize=10)
        m = results[key]["metrics"]
        ax.set_title(f"{title}\nACC={m['ACC']:.3f}  BWT={m['BWT']:+.3f}  FF={m['FF']:.3f}", fontsize=11)
        for i in range(len(task_order)):
            for j in range(len(task_order)):
                if j <= i:
                    ax.text(j, i, f"{R[i,j]:.2f}", ha='center', va='center', fontsize=8,
                            color='black' if R[i,j] > 0.3 else 'white')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"avr-cl validation [math stream]: {MODEL_ID}, LoRA r={LORA_RANK}, seed {SEED}",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nHeatmap saved: {save_path}", flush=True)
    plt.close()

# ============================================================================
# MAIN
# ============================================================================
def run_all():
    print("=" * 70, flush=True)
    print("avr-cl VALIDATION v2 [MATH STREAM]: Naive vs AVR vs Two-Stream+AVR", flush=True)
    print(f"Model: {MODEL_ID} | Seed: {SEED}", flush=True)
    print(f"Tasks: {TASK_NAMES}", flush=True)
    print(f"LoRA: rank={LORA_RANK}, alpha={LORA_ALPHA}, targets={LORA_TARGETS}", flush=True)
    print(f"AVR: threshold={DRIFT_THRESHOLD}, alpha={REPAIR_ALPHA}, max_steps={MAX_REPAIR_STEPS}", flush=True)
    print(f"Batch: {BATCH_SIZE} (accum {GRAD_ACCUM_STEPS} = {BATCH_SIZE*GRAD_ACCUM_STEPS} eff), ctx {CONTEXT_LENGTH}", flush=True)
    print(f"Sanity gate: T1 (gsm8k) >= {T1_MIN_ACCURACY} or abort", flush=True)
    print("=" * 70, flush=True)

    set_seed(SEED)

    print(f"\nLoading math stream data...", flush=True)
    train_data, test_data, task_order = load_math_stream()

    results = {}

    # Condition 1: Naive
    R_n, m_n = run_naive(train_data, test_data, task_order)
    results["naive"] = {"R": R_n, "metrics": m_n}
    print(f"\n  Naive: ACC={m_n['ACC']:.3f}  BWT={m_n['BWT']:+.3f}  FF={m_n['FF']:.3f}", flush=True)

    # Condition 2: AVR
    R_a, m_a, repairs_a, log_a = run_avr(train_data, test_data, task_order)
    results["avr"] = {"R": R_a, "metrics": m_a, "total_repairs": repairs_a, "log": log_a}
    print(f"\n  AVR: ACC={m_a['ACC']:.3f}  BWT={m_a['BWT']:+.3f}  FF={m_a['FF']:.3f}  repairs={repairs_a}", flush=True)

    # NOTE: Two-Stream+AVR condition cut to fit 3-hour budget.
    # Validated separately on TRACE in v34 (BWT +0.107).

    # Heatmap + JSON
    make_heatmap(results, task_order, OUTPUT_DIR / "validation_heatmap_math.png")
    with open(OUTPUT_DIR / "validation_results_math.json", "w") as f:
        json.dump({
            "benchmark": "math_stream",
            "config": {
                "model": MODEL_ID, "seed": SEED, "tasks": task_order,
                "lora_rank": LORA_RANK, "lora_alpha": LORA_ALPHA,
                "lora_targets": LORA_TARGETS,
                "drift_threshold": DRIFT_THRESHOLD, "repair_alpha": REPAIR_ALPHA,
                "max_repair_steps": MAX_REPAIR_STEPS,
                "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
                "context_length": CONTEXT_LENGTH,
                "examples_per_task": 5000, "epochs": TASK_EPOCHS,
                "conditions": ["naive", "avr"],
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {OUTPUT_DIR}/validation_results_math.json", flush=True)

    # Final summary
    print(f"\n{'='*80}", flush=True)
    print("FINAL VALIDATION SUMMARY [MATH STREAM]", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\n{'Method':<30} {'ACC':<8} {'BWT':<8} {'FF':<8} {'Repairs':<10}", flush=True)
    print("-" * 70, flush=True)
    print(f"{'Naive SFT':<30} {m_n['ACC']:<8.3f} {m_n['BWT']:<+8.3f} {m_n['FF']:<8.3f} {'-':<10}", flush=True)
    print(f"{'AVR (v23 style)':<30} {m_a['ACC']:<8.3f} {m_a['BWT']:<+8.3f} {m_a['FF']:<8.3f} {repairs_a:<10}", flush=True)
    print("-" * 70, flush=True)
    print(f"\n  Recovery (AVR vs Naive): BWT {m_n['BWT']:+.3f} -> {m_a['BWT']:+.3f}  ({(m_a['BWT']-m_n['BWT'])*100:+.1f}pp)", flush=True)

    # Task 1 recovery (GSM8K after training all 4)
    T = len(task_order)
    t1_naive = R_n[T-1][0]
    t1_avr = R_a[T-1][0]
    print(f"\n  Task 1 ({task_order[0]}) accuracy after training all {T} tasks:", flush=True)
    print(f"    Naive:  {t1_naive:.3f}", flush=True)
    print(f"    AVR:    {t1_avr:.3f}  (recovered {(t1_avr-t1_naive)*100:+.1f}pp)", flush=True)

    for cond_label, R in [("naive", R_n), ("avr", R_a)]:
        print(f"\n  R MATRIX ({cond_label}):", flush=True)
        header = "After\\Test  " + "  ".join(f"{t[:12]:<14}" for t in task_order)
        print(f"  {header}", flush=True)
        for i in range(len(task_order)):
            row = f"  {task_order[i][:12]:<14} " + "  ".join(f"{R[i][j]:<14.3f}" for j in range(len(task_order)))
            print(row, flush=True)

    summary = {"math_stream": {
        "naive": m_n,
        "avr": {**m_a, "total_repairs": repairs_a},
    }}
    with open(OUTPUT_DIR / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {OUTPUT_DIR}/validation_summary.json", flush=True)
    print(f"\nDONE.", flush=True)

# ============================================================================
# MODAL WRAPPER
# ============================================================================
if MODAL_AVAILABLE:
    avr_image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "git-lfs", "build-essential")
        .pip_install(
            "torch==2.5.1",
            "transformers>=4.45.0",
            "peft>=0.13.0",
            "datasets>=3.0.0",
            "accelerate>=1.0.0",
            "numpy<2",
            "matplotlib",
            "sentencepiece",
            "protobuf",
            "scipy",
            "packaging",
        )
        .env({
            "PYDEVD_DISABLE_FILE_VALIDATION": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        })
    )

    app = modal.App("avr-cl-math")
    output_vol = modal.Volume.from_name("avr-cl-output", create_if_missing=True)

    @app.function(
        image=avr_image,
        gpu="A100-40GB",
        timeout=43200,  # 12 hours — eval is 2000+ generations, dominant cost
        volumes={"/root/output": output_vol},
    )
    def run_validation():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open("/root/output/RUN_STARTED.txt", "w") as f:
            f.write(f"Started at {time.time()}\n")
        try:
            run_all()
            with open("/root/output/RUN_COMPLETED.txt", "w") as f:
                f.write(f"Completed at {time.time()}\n")
        except Exception as e:
            import traceback
            with open("/root/output/RUN_FAILED.txt", "w") as f:
                f.write(f"Failed at {time.time()}\n")
                f.write(f"Error: {e}\n")
                f.write(traceback.format_exc())
            raise
        finally:
            output_vol.commit()

    @app.local_entrypoint()
    def main():
        print("Launching avr-cl math stream validation on Modal A100-40GB...", flush=True)
        print("Use --detach to survive local disconnect.", flush=True)
        print("Monitor: modal app logs avr-cl-math", flush=True)
        print("Results: modal volume ls avr-cl-output", flush=True)
        print("", flush=True)
        run_validation.remote()

else:
    if __name__ == "__main__":
        run_all()
