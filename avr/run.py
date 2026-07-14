"""
avr.run — the orchestrator.

Wires LEARN → VERIFY → REPAIR across a task stream.
Each phase is swappable: import from avr.learn, avr.verify, avr.repair
and pass your own implementations if needed.
"""
import torch
import numpy as np
import gc, copy, random, time
from typing import List, Tuple

from .model import load_model
from .learn import train_sft, consolidate
from .verify import eval_ppls, check_drift
from .repair import get_lora_state, set_lora_state, reset_lora, repair
from .eval import evaluate, default_scorer


def compute_metrics(R, task_order):
    T = len(task_order)
    acc = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]
    bwt = float(np.mean(bwt_values)) if bwt_values else 0.0
    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j] for j in range(T-1)]
    ff = float(np.mean(ff_values)) if ff_values else 0.0
    return {"acc": acc, "bwt": bwt, "ff": ff}


def run(model: str,
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
        repair_fn=None,
        device: str = "cuda",
        seed: int = 42):
    """
    Run Anchor-Verify-Repair on a model + task stream.

    Args:
        model: HuggingFace model ID
        tasks: List of (name, train_examples, eval_examples).
               Each example is (question, answer, gold).
        lora_rank: LoRA rank. Default 128.
        lora_targets: LoRA target modules. Auto-detected if None.
        two_stream: Use hippocampus/neocortex variant. Default False.
        scorer: Custom scorer(response, gold) -> float. Default: substring match.
        repair_fn: Custom repair operator with signature
                   fn(model, snapshot, alpha, device) -> int (num params touched).
                   Default: avr.repair.repair (linear interpolation toward snapshot).
                   Pass your own for TIES, TaskArithmetic, etc.
    """
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    task_order = [t[0] for t in tasks]
    tasks_data = {t[0]: {"train": t[1], "eval": t[2]} for t in tasks}
    T = len(task_order)

    _do_repair = repair_fn if repair_fn is not None else repair

    print(f"\n{'='*70}", flush=True)
    print(f"avr-cl | Model: {model} | Tasks: {task_order}", flush=True)
    print(f"LoRA r={lora_rank} | Two-stream: {two_stream} | Seed: {seed}", flush=True)
    print(f"AVR: threshold={drift_threshold}, alpha={repair_alpha}, max_steps={max_repair_steps}", flush=True)
    print(f"Repair operator: {_do_repair.__name__ if hasattr(_do_repair, '__name__') else 'custom'}", flush=True)
    print(f"{'='*70}", flush=True)

    model_obj, tokenizer = load_model(model, lora_rank, lora_alpha, lora_targets, device)
    R = [[0.0]*T for _ in range(T)]
    best_ppls = {}
    completed = []
    total_repairs = 0
    repair_log = []
    snapshot = None

    if two_stream:
        neo_state = get_lora_state(model_obj)

    for ti, task in enumerate(task_order):
        print(f"\n{'='*60}\n  Task {ti+1}/{T}: {task}\n{'='*60}", flush=True)
        train_ex = tasks_data[task]["train"]

        # LEARN
        if two_stream:
            neo_snapshot = copy.deepcopy(neo_state)
            print(f"  [twostream] Snapshot taken", flush=True)
            reset_lora(model_obj)
            print(f"  [twostream] Hippocampus reset", flush=True)
            train_sft(model_obj, tokenizer, train_ex, epochs=epochs, lr=lr,
                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,
                     device=device, tag="hippo")
            hippo_state = get_lora_state(model_obj)
            print(f"  [twostream] Hippocampus trained", flush=True)
            set_lora_state(model_obj, neo_state, device)
            neo_state = consolidate(model_obj, tokenizer, hippo_state, neo_state, train_ex,
                                    epochs=1, lr=lr*0.5, batch_size=batch_size,
                                    grad_accum=grad_accum, ctx_len=ctx_len, device=device)
            print(f"  [twostream] Consolidation complete", flush=True)
            set_lora_state(model_obj, neo_state, device)
            repair_target = neo_snapshot
        else:
            train_sft(model_obj, tokenizer, train_ex, epochs=epochs, lr=lr,
                     batch_size=batch_size, grad_accum=grad_accum, ctx_len=ctx_len,
                     device=device, tag="sft")
            repair_target = snapshot

        # VERIFY
        post_ppls = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)
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
                    n = _do_repair(model_obj, repair_target, repair_alpha, device)
                    repairs += 1
                    rp = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)
                    still = check_drift(rp, best_ppls, completed, drift_threshold)
                    print(f"    [AVR] Repair {step+1}: {n} params, drifted: {list(still.keys()) if still else 'none'}", flush=True)
                    if not still:
                        print(f"  [AVR] Converged at step {step+1}", flush=True)
                        break
                if still:
                    print(f"  [AVR] Max steps reached", flush=True)
                if two_stream:
                    neo_state = get_lora_state(model_obj)
            else:
                print(f"  [AVR] No drift", flush=True)

        total_repairs += repairs
        repair_log.append({"task": task, "repairs": repairs})

        # Update best PPLs
        final_ppls = eval_ppls(model_obj, tokenizer, tasks_data, task_order, ti+1, device)
        for dk, dp in final_ppls.items():
            if dk not in best_ppls or dp < best_ppls[dk]:
                best_ppls[dk] = dp

        # Snapshot
        if not two_stream:
            snapshot = get_lora_state(model_obj)
        completed.append(task)

        # Evaluate (R-matrix)
        print(f"\n  Evaluating...", flush=True)
        for j in range(ti + 1):
            R[ti][j] = evaluate(model_obj, tokenizer, tasks_data[task_order[j]]["eval"],
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

    del model_obj; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result
