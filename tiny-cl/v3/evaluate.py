"""
V3 Evaluation: Perplexity + health scores.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List


@torch.no_grad()
def compute_perplexity(model, dataset, device: str = "cuda",
                       max_samples: int = 2048) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    total_loss = 0.0
    n_tokens = 0

    for batch in loader:
        if n_tokens >= max_samples * 512:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item() * input_ids.numel()
        n_tokens += input_ids.numel()

    model.train()
    if n_tokens == 0:
        return float("inf")
    avg_loss = total_loss / n_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def evaluate_all_phases(
    model,
    val_datasets: Dict,
    completed_phases: List[str],
    device: str = "cuda",
    max_samples: int = 2048,
) -> Dict:
    results = {"perplexity": {}}
    for phase_key in completed_phases:
        if phase_key in val_datasets:
            ppl = compute_perplexity(model, val_datasets[phase_key], device, max_samples)
            results["perplexity"][phase_key] = ppl
    return results


def compute_forgetting_factors(all_results: Dict, domain_keys: List[str]) -> Dict:
    forgetting = {}
    last_phase_key = list(all_results["phases"].keys())[-1]
    final_ppls = all_results["phases"][last_phase_key].get("perplexity", {})
    for pk in domain_keys:
        after_learn_ppl = all_results["phases"][pk].get("perplexity", {}).get(pk, None)
        after_all_ppl = final_ppls.get(pk, None)
        if after_learn_ppl and after_all_ppl and after_learn_ppl > 0:
            forgetting[pk] = after_all_ppl / after_learn_ppl
    return forgetting
