"""
V2 Evaluation: Perplexity, Forgetting Factor, BWT.
Evaluates a model on held-out data from all seen phases.
"""

import math
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional


@torch.no_grad()
def compute_perplexity(
    model,
    dataset,
    device: str = "cuda",
    batch_size: int = 16,
    max_samples: int = 2048,
) -> float:
    """
    Compute perplexity on a dataset.
    Returns perplexity (lower = better).
    """
    model.eval()

    # Limit samples for speed
    n_samples = min(len(dataset), max_samples)
    if n_samples == 0:
        return float('inf')

    # Create dataloader without dropping last
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for i, batch in enumerate(dataloader):
        if n_batches * batch_size >= max_samples:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)

        # Scale loss by number of tokens in batch
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
        n_batches += 1

    if total_tokens == 0:
        return float('inf')

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()
    return perplexity


def evaluate_all_phases(
    model,
    val_datasets: Dict[str, object],
    completed_phases: List[str],
    device: str = "cuda",
    max_samples: int = 2048,
) -> Dict:
    """
    Evaluate model on all completed phases.
    Returns {perplexity: {phase: ppl}, bwt: float}
    """
    results = {"perplexity": {}}

    for phase_key in completed_phases:
        if phase_key not in val_datasets:
            continue

        ppl = compute_perplexity(
            model,
            val_datasets[phase_key],
            device=device,
            max_samples=max_samples,
        )
        results["perplexity"][phase_key] = ppl
        print(f"    Eval Phase {phase_key}: ppl = {ppl:.2f}")

    return results


def compute_bwt(phase_perplexities: Dict[str, Dict[str, float]]) -> float:
    """
    Compute Backward Transfer (BWT).

    BWT = average change in performance on old tasks after learning new tasks.
    BWT < 0 means forgetting.
    BWT > 0 means backward improvement.

    Args:
        phase_perplexities: {phase_key: {eval_phase: ppl}}
            For each training phase, the perplexity on all eval phases.

    Returns:
        BWT value (negative = forgetting)
    """
    phase_keys = list(phase_perplexities.keys())
    if len(phase_keys) < 2:
        return 0.0

    bwt_values = []

    for i, phase_key in enumerate(phase_keys[:-1]):
        # Perplexity on this phase right after learning it
        after_learn = phase_perplexities[phase_key].get(phase_key, None)
        # Perplexity on this phase after all subsequent phases
        after_all = phase_perplexities[phase_keys[-1]].get(phase_key, None)

        if after_learn is not None and after_all is not None and after_learn > 0:
            # For perplexity: increase = forgetting, so BWT = -(after_all - after_learn)
            bwt = -(after_all - after_learn) / after_learn
            bwt_values.append(bwt)

    return sum(bwt_values) / len(bwt_values) if bwt_values else 0.0


def compute_forgetting_factor(phase_perplexities: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compute Forgetting Factor for each phase.

    Forgetting Factor = final_ppl / after_learn_ppl
    1.0 = no forgetting, >1.0 = forgot some, <1.0 = improved

    Returns:
        {phase_key: forgetting_factor}
    """
    phase_keys = list(phase_perplexities.keys())
    forgetting = {}

    for phase_key in phase_keys:
        after_learn = phase_perplexities[phase_key].get(phase_key, None)
        after_all = phase_perplexities[phase_keys[-1]].get(phase_key, None)

        if after_learn is not None and after_all is not None and after_learn > 0:
            forgetting[phase_key] = after_all / after_learn

    return forgetting
