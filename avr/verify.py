"""VERIFY phase: PPL drift detection."""
import math
import torch
from .model import format_example


def compute_ppl(model, tokenizer, examples, device="cuda", max_samples=100):
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
