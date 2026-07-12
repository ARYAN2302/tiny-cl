"""
avr.run — the main entry point.

Usage:
    import avr
    result = avr.run(
        model="Qwen/Qwen3-1.7B",
        tasks=[
            ("gsm8k", train_pairs, eval_pairs),
            ("math", train_pairs, eval_pairs),
        ],
        lora_rank=128,
    )
    print(result["bwt"], result["acc"], result["repairs"])

Or via CLI:
    avr train config.yaml
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import json, math, gc, re, time, copy, random
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(model_id: str, lora_rank: int = 128, lora_alpha: int = 128,
               lora_targets: list = None, device: str = "cuda"):
    """Load any HF model + LoRA. Auto-detects LoRA targets if not specified."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager")

    # Auto-detect LoRA targets if not specified
    if lora_targets is None:
        lora_targets = _detect_lora_targets(model)
        print(f"  Auto-detected LoRA targets: {lora_targets}", flush=True)

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=lora_targets, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    return model, tokenizer


def _detect_lora_targets(model):
    """Detect available LoRA target modules from model architecture."""
    target_candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "in_proj", "out_proj", "conv1d"]
    found = []
    for name, _ in model.named_modules():
        for candidate in target_candidates:
            if name.endswith(candidate) and candidate not in found:
                found.append(candidate)
    return found if found else ["q_proj", "v_proj"]


# ============================================================================
# CHAT TEMPLATE WRAPPING
# ============================================================================
def format_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: no chat template
        return f"User: {question}\nAssistant:"


def format_example(tokenizer, question: str, answer: str) -> str:
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = f"User: {question}\nAssistant: {answer}"
    return text + tokenizer.eos_token


# ============================================================================
# LoRA STATE
# ============================================================================
def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state, device="cuda"):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device).to(p.data.dtype))

def reset_lora(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()


# ============================================================================
# AVR CORE
# ============================================================================
def compute_ppl(model, tokenizer, examples, device="cuda", max_samples=100):
    """examples = list of (question, answer, gold). Uses question + answer for PPL."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for question, answer, gold in examples[:max_samples]:
        text = format_example(tokenizer, question, answer)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def eval_ppls(model, tokenizer, tasks_data, task_order, trained_so_far, device="cuda"):
    ppls = {}
    for i, task in enumerate(task_order):
        if i >= trained_so_far:
            break
        ppls[task] = compute_ppl(model, tokenizer, tasks_data[task]["train"], device=device)
    return ppls


def check_drift(current_ppls, best_ppls, completed_tasks, threshold=1.15):
    drifted = {}
    for task in completed_tasks:
        if task not in current_ppls or task not in best_ppls:
            continue
        ratio = current_ppls[task] / best_ppls[task] if best_ppls[task] > 0 else 1.0
        if ratio > threshold:
            drifted[task] = {"current": current_ppls[task], "best": best_ppls[task], "ratio": ratio}
    return drifted


def repair(model, snapshot, alpha=0.1, device="cuda"):
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name and name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))
            n += 1
    return n


# ============================================================================
# TRAINING
# ============================================================================
from torch.utils.data import Dataset, DataLoader


class _TextDataset(Dataset):
    def __init__(self, token_ids, ctx_len):
        self.token_ids = token_ids
        self.ctx_len = ctx_len
        self.n_chunks = max(1, len(token_ids) // ctx_len)
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        s = idx * self.ctx_len
        e = s + self.ctx_len
        chunk = self.token_ids[s:e]
        return {"input_ids": chunk, "labels": chunk.clone()}


def _build_tokens(tokenizer, examples):
    """examples = list of (question, answer, gold)."""
    all_tokens = []
    for question, answer, gold in examples:
        text = format_example(tokenizer, question, answer)
        all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
    return torch.tensor(all_tokens, dtype=torch.long)


def train_sft(model, tokenizer, examples, epochs=3, lr=2e-4, batch_size=4,
              grad_accum=4, ctx_len=512, device="cuda", tag="sft"):
    token_ids = _build_tokens(tokenizer, examples)
    dataset = _TextDataset(token_ids, ctx_len)
    print(f"    [{tag}] {len(token_ids):,} tokens, {len(dataset)} chunks", flush=True)

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            model.train()
            out = model(input_ids=batch["input_ids"].to(device),
                       labels=batch["labels"].to(device))
            (out.loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += out.loss.item()
                if gs % 50 == 0:
                    print(f"      [{tag}] step {gs} | loss={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)


def consolidate(model, tokenizer, hippo_state, neo_state, examples,
                epochs=1, lr=1e-4, batch_size=4, grad_accum=4, ctx_len=512,
                device="cuda"):
    """KL distill hippocampus -> neocortex (two-stream consolidation)."""
    print(f"    [consolid] KL distill ({epochs} epoch)", flush=True)
    token_ids = _build_tokens(tokenizer, examples)
    dataset = _TextDataset(token_ids, ctx_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for n, p in model.named_parameters():
        if "lora_" in n: p.requires_grad = True
        else: p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    gs, tl = 0, 0.0
    t0 = time.time()
    accum = 0
    opt.zero_grad()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            set_lora_state(model, hippo_state, device)
            model.eval()
            with torch.no_grad():
                hippo_logits = model(input_ids=input_ids).logits
                p_hippo = F.softmax(hippo_logits[..., :-1, :].contiguous().float(), dim=-1)
            del hippo_logits

            set_lora_state(model, neo_state, device)
            model.train()
            neo_logits = model(input_ids=input_ids).logits
            shift_neo = neo_logits[..., :-1, :].contiguous()
            log_p_neo = F.log_softmax(shift_neo.float(), dim=-1)
            kl_loss = F.kl_div(log_p_neo, p_hippo, reduction='batchmean', log_target=False)
            (kl_loss / grad_accum).backward()
            accum += 1
            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                accum = 0
                gs += 1
                tl += kl_loss.item()
                if gs % 50 == 0:
                    print(f"      [consolid] step {gs} | KL={tl/gs:.4f} | {time.time()-t0:.0f}s", flush=True)
            del neo_logits, log_p_neo, p_hippo, kl_loss, shift_neo
            neo_state = get_lora_state(model)
    print(f"    [consolid] Done: {gs} steps, avg KL={tl/max(gs,1):.4f}", flush=True)
    return neo_state


# ============================================================================
# EVALUATION (batched)
# ============================================================================
def generate_batch(model, tokenizer, questions, max_new_tokens=200,
                   batch_size=8, device="cuda"):
    results = []
    gc_was = getattr(model, "gradient_checkpointing", False)
    if gc_was:
        try: model.gradient_checkpointing_disable()
        except: pass
    model.eval()
    try:
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            texts = [format_prompt(tokenizer, q) for q in batch]
            tokenizer.padding_side = "left"
            inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                             max_length=1024, padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=tokenizer.pad_token_id, temperature=1.0)
            for out in outputs:
                input_len = inputs["input_ids"].shape[1]
                results.append(tokenizer.decode(out[input_len:],
                    skip_special_tokens=True).strip())
    finally:
        if gc_was:
            try: model.gradient_checkpointing_enable(); model.enable_input_require_grads()
            except: pass
    return results


def normalize_answer(s):
    s = s.strip().lower()
    s = re.sub(r'[^\w\s.-]', ' ', s)
    return ' '.join(s.split())


def default_scorer(response, gold):
    """Default scorer: normalized substring match."""
    resp = normalize_answer(response)
    g = normalize_answer(gold)
    if resp == g: return 1.0
    if g in resp or resp in g: return 1.0
    g_spaces = g.replace('_', ' ')
    resp_spaces = resp.replace('_', ' ')
    if g_spaces in resp_spaces or resp_spaces in g_spaces: return 1.0
    return 0.0


def evaluate(model, tokenizer, eval_examples, task_name, scorer=None,
             max_questions=200, batch_size=8, device="cuda"):
    """eval_examples = list of (question, answer, gold). Uses question + gold."""
    if scorer is None:
        scorer = default_scorer
    total = min(len(eval_examples), max_questions)
    examples = eval_examples[:total]
    questions = [ex[0] for ex in examples]
    golds = [ex[2] for ex in examples]

    print(f"    Eval {task_name} ({total} Qs)...", flush=True)
    correct = 0
    t0 = time.time()
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i:i+batch_size]
        batch_g = golds[i:i+batch_size]
        responses = generate_batch(model, tokenizer, batch_q,
                                   max_new_tokens=200, batch_size=len(batch_q),
                                   device=device)
        for r, g in zip(responses, batch_g):
            if scorer(r, g): correct += 1
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return acc


# ============================================================================
# METRICS
# ============================================================================
def compute_metrics(R, task_order):
    T = len(task_order)
    acc = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]
    bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]
    ff = float(np.mean(ff_values)) if ff_values else 0.0
    return {"acc": acc, "bwt": bwt, "ff": ff}


# ============================================================================
# THE MAIN LOOP
# ============================================================================
def run(model_id: str,
        tasks: List[Tuple[str, list, list]],
        lora_rank: int = 128,
        lora_alpha: int = 128,
        lora_targets: list = None,
        epochs: int = 3,
        lr: float = 2e-4,
        batch_size: int = 4,
        grad_accum: int = 4,
        ctx_len: int = 512,
        drift_threshold: float = 1.15,
        repair_alpha: float = 0.1,
        max_repair_steps: int = 10,
        two_stream: bool = False,
        scorer=None,
        device: str = "cuda",
        seed: int = 42):
    """
    Run AVR continual learning on a model + task stream.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-1.7B")
        tasks: List of (name, train_examples, eval_examples).
               Each example is a (question, answer, gold) tuple.
               - question: the input prompt
               - answer: the full training target (reasoning + answer)
               - gold: the short gold answer for scoring
        lora_rank: LoRA rank (default 128)
        lora_targets: LoRA target modules. Auto-detected if None.
        two_stream: Use two-stream (hippocampus/neocortex) variant. Default False.
        scorer: Custom scoring function(response, gold) -> float. Default: substring match.
        device: "cuda" or "cpu"

    Returns:
        Dict with keys: acc, bwt, ff, repairs, R (R-matrix), repair_log
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    task_order = [t[0] for t in tasks]
    tasks_data = {t[0]: {"train": t[1], "eval": t[2]} for t in tasks}
    T = len(task_order)

    print(f"\n{'='*70}", flush=True)
    print(f"avr-cl | Model: {model_id} | Tasks: {task_order}", flush=True)
    print(f"LoRA r={lora_rank} | Two-stream: {two_stream} | Seed: {seed}", flush=True)
    print(f"AVR: threshold={drift_threshold}, alpha={repair_alpha}, max_steps={max_repair_steps}", flush=True)
    print(f"{'='*70}", flush=True)

    model, tokenizer = load_model(model_id, lora_rank, lora_alpha, lora_targets, device)
    R = [[0.0]*T for _ in range(T)]
    best_ppls = {}
    completed = []
    total_repairs = 0
    repair_log = []
    snapshot = None

    if two_stream:
        neo_state = get_lora_state(model)

    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task}\n{'='*60}", flush=True)
        train_ex = tasks_data[task]["train"]

        if two_stream:
            # Two-stream: hippocampus + consolidation + AVR on neocortex
            neo_snapshot = copy.deepcopy(neo_state)
            print(f"  [twostream] Snapshot taken", flush=True)
            reset_lora(model)
            print(f"  [twostream] Hippocampus reset", flush=True)
            train_sft(model, tokenizer, train_ex, epochs=epochs, lr=lr,
                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,
                     device=device, tag="hippo")
            hippo_state = get_lora_state(model)
            print(f"  [twostream] Hippocampus trained", flush=True)
            set_lora_state(model, neo_state, device)
            neo_state = consolidate(model, tokenizer, hippo_state, neo_state, train_ex,
                                    epochs=1, lr=lr*0.5, batch_size=batch_size,
                                    grad_accum=grad_accum, ctx_len=ctx_len, device=device)
            print(f"  [twostream] Consolidation complete", flush=True)
            set_lora_state(model, neo_state, device)
            repair_target = neo_snapshot
        else:
            # Standard AVR: SFT + verify + repair
            train_sft(model, tokenizer, train_ex, epochs=epochs, lr=lr,
                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,
                     device=device, tag="sft")
            repair_target = snapshot

        # VERIFY
        post_ppls = eval_ppls(model, tokenizer, tasks_data, task_order, ti+1, device)
        if task not in best_ppls:
            best_ppls[task] = post_ppls[task]
        print(f"  PPLs: " + " | ".join(f"{k}:{v:.2f}" for k,v in post_ppls.items()), flush=True)

        # REPAIR
        repairs = 0
        if ti > 0 and repair_target is not None:
            drifted = check_drift(post_ppls, best_ppls, completed, drift_threshold)
            if drifted:
                print(f"  [AVR] DRIFT on {list(drifted.keys())}", flush=True)
                for dk, info in drifted.items():
                    print(f"    {dk}: {info['current']:.2f}/{info['best']:.2f}={info['ratio']:.2f}x", flush=True)
                still = drifted
                for step in range(max_repair_steps):
                    n = repair(model, repair_target, repair_alpha, device)
                    repairs += 1
                    rp = eval_ppls(model, tokenizer, tasks_data, task_order, ti+1, device)
                    still = check_drift(rp, best_ppls, completed, drift_threshold)
                    print(f"    [AVR] Repair {step+1}: {n} params, drifted: {list(still.keys()) if still else 'none'}", flush=True)
                    if not still:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still:
                    print(f"  [AVR] Max steps reached", flush=True)
                if two_stream:
                    neo_state = get_lora_state(model)
            else:
                print(f"  [AVR] No drift", flush=True)

        total_repairs += repairs
        repair_log.append({"task": task, "repairs": repairs})

        # Update best PPLs
        final_ppls = eval_ppls(model, tokenizer, tasks_data, task_order, ti+1, device)
        for dk, dp in final_ppls.items():
            if dk not in best_ppls or dp < best_ppls[dk]:
                best_ppls[dk] = dp

        # Snapshot for next task
        if not two_stream:
            snapshot = get_lora_state(model)
        completed.append(task)

        # Evaluate on all tasks seen so far (R-matrix)
        print(f"\n  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate(model, tokenizer, tasks_data[task_order[j]]["eval"],
                               task_order[j], scorer=scorer, device=device)

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    metrics = compute_metrics(R, task_order)
    result = {
        "acc": metrics["acc"],
        "bwt": metrics["bwt"],
        "ff": metrics["ff"],
        "repairs": total_repairs,
        "R": R,
        "repair_log": repair_log,
        "task_order": task_order,
    }

    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS: ACC={metrics['acc']:.3f}  BWT={metrics['bwt']:+.3f}  FF={metrics['ff']:.3f}  Repairs={total_repairs}", flush=True)
    print(f"{'='*70}", flush=True)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result
