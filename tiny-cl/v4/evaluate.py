"""
V4 Evaluation: Perplexity, forgetting factors, streaming metrics.
"""

import math
import torch
from torch.utils.data import DataLoader
from typing import Dict, List


@torch.no_grad()
def compute_perplexity(model, dataset, device="cpu", max_samples=512):
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in loader:
        if n_batches * 16 >= max_samples:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        n_tokens = (labels != 0).sum().item()  # Count non-pad tokens
        if n_tokens == 0:
            n_tokens = input_ids.numel()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1

    model.train()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def evaluate_all_phases(model, val_datasets, completed_phases, device="cpu",
                        max_samples=512):
    results = {"perplexity": {}}
    for pk in completed_phases:
        if pk in val_datasets:
            ppl = compute_perplexity(model, val_datasets[pk], device, max_samples)
            results["perplexity"][pk] = ppl
    return results


def compute_forgetting_factors(all_results, domain_keys):
    """Final PPL / after-learn PPL per domain."""
    forgetting = {}
    last_phase_key = list(all_results["phases"].keys())[-1]
    final_ppls = all_results["phases"][last_phase_key].get("perplexity", {})

    for pk in domain_keys:
        after_learn = all_results["phases"][pk].get("perplexity", {}).get(pk)
        after_all = final_ppls.get(pk)
        if after_learn and after_all and after_learn > 0:
            forgetting[pk] = after_all / after_learn
    return forgetting
