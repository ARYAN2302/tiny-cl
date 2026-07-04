"""
avr.metrics — R-matrix, BWT, FF, ACC.

Standard continual-learning metrics. The R matrix is R[i][j] = accuracy on
task j after training task i. From it we derive:
    ACC = mean of last row
    BWT = mean(R[T-1][j] - R[j][j] for j < T-1)   — backward transfer
    FF  = mean(max(R[l][j] for l) - R[T-1][j])     — forgetting factor
"""

from __future__ import annotations
from typing import List, Dict, Any
import re
import math
import torch
import numpy as np


def generate(model, tokenizer, prompt, max_new_tokens=20, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True).strip()


def score_answer(response: str, gold: str) -> float:
    """Score a response against gold. Handles MCQ, numeric, exact-match."""
    response = response.strip()
    gold = gold.strip()
    # MCQ: A/B/C/D/E
    if gold in ["A", "B", "C", "D", "E"]:
        response_upper = response.upper()[:5]
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in response_upper:
                return 1.0 if letter == gold else 0.0
        return 0.0
    # Numeric
    if re.match(r'^[\d.-]+', gold):
        numbers = re.findall(r'[\d.]+', response)
        if numbers:
            return 1.0 if numbers[-1] == gold else 0.0
        return 0.0
    # Exact match (normalized)
    def norm(s):
        s = s.lower().strip()
        s = re.sub(r'[^\w\s.-]', ' ', s)
        return ' '.join(s.split())
    return 1.0 if norm(response) == norm(gold) else 0.0


def evaluate_task_accuracy(model, tokenizer, test_pairs, task_name,
                           max_questions=200, device="cuda"):
    """Accuracy on a task's test set."""
    print(f"    Eval {task_name} ({min(len(test_pairs), max_questions)} Qs)...")
    correct = 0
    total = min(len(test_pairs), max_questions)
    for i in range(total):
        prompt, gold = test_pairs[i]
        response = generate(model, tokenizer, prompt,
                            max_new_tokens=20, device=device)
        if score_answer(response, gold):
            correct += 1
    acc = correct / total
    print(f"    {task_name}: {correct}/{total} = {acc:.3f}")
    return acc


def compute_metrics(R, task_order):
    """ACC, BWT, FF from the R matrix."""
    T = len(task_order)
    ACC = float(np.mean([R[T-1][j] for j in range(T)]))
    bwt_values = [R[T-1][j] - R[j][j] for j in range(T-1)]
    BWT = float(np.mean(bwt_values)) if bwt_values else 0.0
    ff_values = [max(R[l][j] for l in range(T)) - R[T-1][j]
                 for j in range(T-1)]
    FF = float(np.mean(ff_values)) if ff_values else 0.0
    return {"ACC": ACC, "BWT": BWT, "FF": FF}


def make_r_matrix_callback(R, task_order, device="cuda"):
    """Build a callback that fills the R matrix as tasks complete.

    The callback is called after each task with (model, tokenizer, state, i).
    It evaluates accuracy on ALL tasks seen so far and fills R[i][j].
    """
    T = len(task_order)

    def callback(model, tokenizer, state, i):
        for j in range(i + 1):
            R[i][j] = evaluate_task_accuracy(
                model, tokenizer,
                # task_order[j]'s test pairs are in TaskSpec —
                # but the callback doesn't have them. We need to pass them.
                # HACK: stash them on the callback closure.
                callback._test_pairs[j],
                task_order[j],
                device=device,
            )
    return callback
