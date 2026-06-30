"""
Evaluation: perplexity per phase, BWT, FWT metrics.
"""

import torch
import math
from tqdm import tqdm
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_perplexity(
    model: GPT2LMHeadModel,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: int = None,
    desc: str = "Evaluating",
) -> float:
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs.loss.item()
        total_batches += 1
        
        if max_batches and total_batches >= max_batches:
            break
    
    model.train()
    
    if total_batches == 0:
        return float('inf')
    
    avg_loss = total_loss / total_batches
    perplexity = math.exp(avg_loss)
    return perplexity


def evaluate_all_phases(
    model: GPT2LMHeadModel,
    phases_data: dict,
    eval_phases: list,
    device: str = "cuda",
    max_batches: int = 64,
) -> dict:
    """
    Evaluate model on all specified phases.
    
    Args:
        model: The model to evaluate
        phases_data: Dict with phase data from data.py
        eval_phases: List of phase keys to evaluate on (e.g., ["A", "B", "C"])
        device: Device
        max_batches: Limit eval batches for speed
    
    Returns:
        Dict of phase_key -> perplexity
    """
    results = {}
    
    for phase_key in eval_phases:
        if phase_key not in phases_data:
            continue
        
        val_loader = phases_data[phase_key]["val_loader"]
        ppl = compute_perplexity(
            model, val_loader, device,
            max_batches=max_batches,
            desc=f"Eval Phase {phase_key}",
        )
        results[phase_key] = ppl
    
    return results


def compute_backward_transfer(phase_results: dict, phase_order: list = ["A", "B", "C"]) -> float:
    """
    Compute Backward Transfer (BWT).
    
    BWT = (1/T-1) * Σ (R_T,i - R_i,i) for i < T
    
    Where R_T,i is the performance on task i after learning all T tasks,
    and R_i,i is the performance on task i right after learning it.
    
    Negative BWT = forgetting.
    Positive BWT = backward improvement.
    
    Args:
        phase_results: Dict of {phase_after_which_trained: {phase_evaluated: perplexity}}
        phase_order: Order of phases
    
    Returns:
        BWT score (using negative perplexity as performance, so higher = better)
    """
    # Convert perplexity to "performance" (negative ppl, so lower ppl = higher performance)
    # We use 1/ppl as a simple proxy
    n_phases = len(phase_order)
    
    if n_phases < 2:
        return 0.0
    
    bwt_sum = 0.0
    count = 0
    
    for i in range(n_phases - 1):
        phase_i = phase_order[i]
        
        # Performance on phase i right after learning it
        if phase_i not in phase_results.get(phase_i, {}):
            continue
        perf_after_learning = 1.0 / phase_results[phase_i][phase_i]
        
        # Performance on phase i after learning all subsequent phases
        last_phase = phase_order[-1]
        if phase_i not in phase_results.get(last_phase, {}):
            continue
        perf_after_all = 1.0 / phase_results[last_phase][phase_i]
        
        bwt_sum += (perf_after_all - perf_after_learning)
        count += 1
    
    if count == 0:
        return 0.0
    
    return bwt_sum / count


def compute_forgetting_measure(phase_results: dict, phase_order: list = ["A", "B", "C"]) -> dict:
    """
    Compute Forgetting Measure for each phase.
    
    FM_i = max(0, R_i,i - max_{j>i} R_j,i)
    
    How much the model forgot from its peak performance on each phase.
    
    Args:
        phase_results: Dict of {phase_after_which_trained: {phase_evaluated: perplexity}}
        phase_order: Order of phases
    
    Returns:
        Dict of phase_key -> forgetting measure (in perplexity points)
    """
    forgetting = {}
    
    for i, phase_i in enumerate(phase_order):
        if phase_i not in phase_results.get(phase_i, {}):
            continue
        
        # Best performance on phase i (right after learning it)
        best_ppl = phase_results[phase_i][phase_i]
        
        # Worst performance on phase i after learning subsequent phases
        worst_ppl = best_ppl
        for j in range(i + 1, len(phase_order)):
            phase_j = phase_order[j]
            if phase_i in phase_results.get(phase_j, {}):
                worst_ppl = max(worst_ppl, phase_results[phase_j][phase_i])
        
        # Forgetting = increase in perplexity (worse)
        forgetting[phase_i] = worst_ppl - best_ppl
    
    return forgetting


def format_results_table(
    all_results: dict,
    phase_order: list = ["A", "B", "C"],
) -> str:
    """Format results as a readable table."""
    header = f"{'Method':<25} |"
    for phase in phase_order:
        header += f" Phase {phase} (ppl) |"
    header += f" {'BWT':>8} | {'FM_A':>6} | {'FM_B':>6} | {'Storage':>8} |"
    
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    for method_name, method_data in all_results.items():
        # Get final phase results
        last_phase = phase_order[-1]
        if last_phase not in method_data.get("phase_results", {}):
            continue
        
        final_results = method_data["phase_results"][last_phase]
        display = method_data.get("display_name", method_name)
        
        row = f"{display:<25} |"
        for phase in phase_order:
            ppl = final_results.get(phase, float('inf'))
            row += f" {ppl:>14.1f} |"
        
        # BWT
        bwt = method_data.get("bwt", 0)
        row += f" {bwt:>8.4f} |"
        
        # Forgetting measures
        fm = method_data.get("forgetting", {})
        for phase in phase_order[:-1]:
            fm_val = fm.get(phase, 0)
            row += f" {fm_val:>6.1f} |"
        
        # Storage
        storage = method_data.get("storage_kb", 0)
        if storage > 0:
            row += f" {storage:>6.1f}KB |"
        else:
            row += f" {'0':>8} |"
        
        lines.append(row)
    
    lines.append(separator)
    return "\n".join(lines)
