"""
Main training loop for a single experiment.
Trains a model through all 3 phases with a specified CL method.
"""

import os
import json
import time
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, get_cosine_schedule_with_warmup

from config import ModelConfig, TrainConfig, MethodConfig, PHASE_CONFIGS
from model import create_model, count_parameters, unfreeze_all
from data import get_tokenizer, load_all_phases
from methods import create_method
from evaluate import evaluate_all_phases, compute_backward_transfer, compute_forgetting_measure


def setup_optimizer(model, train_config, method_name):
    """Create optimizer with optional parameter filtering."""
    # Standard AdamW with weight decay on non-bias/non-norm params
    no_decay = ["bias", "ln_", "norm"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": train_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups, lr=train_config.learning_rate)
    return optimizer


def train_single_phase(
    model: GPT2LMHeadModel,
    phase_key: str,
    phase_data: dict,
    method,
    train_config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    scheduler,
    phase_results: dict,
    all_phases_data: dict,
):
    """
    Train the model on a single phase.
    
    Args:
        model: The model
        phase_key: "A", "B", or "C"
        phase_data: DataLoader etc for this phase
        method: CL method instance
        train_config: Training config
        optimizer: Optimizer
        scheduler: LR scheduler
        phase_results: Dict to accumulate results
        all_phases_data: All phase data (for evaluation)
    """
    device = train_config.device
    train_loader = phase_data["train_loader"]
    n_epochs = train_config.epochs_per_phase
    
    print(f"\n{'='*60}")
    print(f"PHASE {phase_key}: Training on {phase_data['word_count']:,} words "
          f"({len(train_loader)} batches/epoch, {n_epochs} epochs)")
    print(f"{'='*60}")
    
    # Method-specific setup before phase
    method.on_phase_start(model, phase_key, all_phases_data)
    
    # Evaluate before training on this phase (for phases we've already learned)
    phases_to_eval = list(phase_results.get("phase_results", {}).keys())
    if phase_key not in phases_to_eval:
        phases_to_eval.append(phase_key)
    
    pre_phase_eval = evaluate_all_phases(
        model, all_phases_data, phases_to_eval, device,
        max_batches=32,
    )
    print(f"  Pre-phase eval: {pre_phase_eval}")
    
    step_count = 0
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Phase {phase_key} Epoch {epoch+1}/{n_epochs}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Compute loss (method-specific)
            loss, metrics = method.compute_loss(
                model,
                {"input_ids": input_ids, "labels": labels},
                device,
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1
            step_count += 1
            
            # Progress bar
            avg_loss = epoch_loss / n_batches
            pbar.set_postfix({"loss": f"{avg_loss:.3f}", "ppl": f"{torch.exp(torch.tensor(avg_loss)):.1f}"})
            
            # Discrete AVR: verify and repair periodically
            if hasattr(method, 'should_verify') and method.should_verify():
                drift_report = method.verify_and_repair(model, device)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        print(f"  Epoch {epoch+1}: loss={avg_epoch_loss:.3f}, "
              f"ppl={torch.exp(torch.tensor(avg_epoch_loss)):.1f} | "
              f"{avg_metrics}")
    
    # Evaluate after training on this phase
    eval_phases = list(phase_results.get("phase_results", {}).keys())
    if phase_key not in eval_phases:
        eval_phases.append(phase_key)
    
    post_phase_eval = evaluate_all_phases(
        model, all_phases_data, eval_phases, device,
        max_batches=64,
    )
    print(f"  Post-phase eval: {post_phase_eval}")
    
    # Store results
    if "phase_results" not in phase_results:
        phase_results["phase_results"] = {}
    phase_results["phase_results"][phase_key] = post_phase_eval
    
    # Method-specific actions after phase
    method.on_phase_end(model, phase_key, phase_data["train_dataset"])


def run_experiment(
    model_config: ModelConfig,
    method_config: MethodConfig,
    train_config: TrainConfig,
    cache_dir: str = None,
) -> dict:
    """
    Run a single experiment: train a model through all 3 phases
    with a specified CL method.
    
    Returns:
        Dict with all results, metrics, and storage info.
    """
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {method_config.display_name} | {model_config.name} model")
    print(f"{'#'*70}")
    
    start_time = time.time()
    device = train_config.device
    
    # Set seed
    torch.manual_seed(train_config.seed)
    
    # Create model
    model = create_model(model_config).to(device)
    
    # Load data
    tokenizer = get_tokenizer()
    phases_data = load_all_phases(PHASE_CONFIGS, tokenizer, train_config, cache_dir)
    
    # Create method
    method = create_method(method_config.name, method_config, train_config)
    
    # Setup optimizer
    optimizer = setup_optimizer(model, train_config, method_config.name)
    
    # Count total training steps for scheduler
    total_steps = 0
    for phase_key in ["A", "B", "C"]:
        total_steps += len(phases_data[phase_key]["train_loader"]) * train_config.epochs_per_phase
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Results tracking
    results = {
        "method": method_config.name,
        "display_name": method_config.display_name,
        "model_size": model_config.name,
        "phase_results": {},
        "forgetting_curves": {},  # phase_key -> list of ppl over time
    }
    
    # Initial evaluation (random model)
    initial_eval = evaluate_all_phases(
        model, phases_data, ["A", "B", "C"], device,
        max_batches=32,
    )
    results["initial_eval"] = initial_eval
    print(f"Initial eval (random model): {initial_eval}")
    
    # Train through phases sequentially
    for phase_key in ["A", "B", "C"]:
        train_single_phase(
            model=model,
            phase_key=phase_key,
            phase_data=phases_data[phase_key],
            method=method,
            train_config=train_config,
            optimizer=optimizer,
            scheduler=scheduler,
            phase_results=results,
            all_phases_data=phases_data,
        )
    
    # Final evaluation on all phases
    final_eval = evaluate_all_phases(
        model, phases_data, ["A", "B", "C"], device,
    )
    results["phase_results"]["final"] = final_eval
    
    # Compute BWT
    results["bwt"] = compute_backward_transfer(results["phase_results"])
    
    # Compute forgetting measures
    results["forgetting"] = compute_forgetting_measure(results["phase_results"])
    
    # Storage info
    if hasattr(method, 'anchor_store'):
        results["storage_kb"] = method.anchor_store.get_storage_size_kb()
    elif hasattr(method, 'buffer'):
        results["storage_kb"] = method.buffer.get_storage_size_kb()
    else:
        results["storage_kb"] = 0
    
    # Parameter count
    results["param_count"] = count_parameters(model)
    
    # Timing
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    print(f"\n✅ Experiment complete in {elapsed/60:.1f} minutes")
    print(f"   Final eval: {final_eval}")
    print(f"   BWT: {results['bwt']:.4f}")
    print(f"   Forgetting: {results['forgetting']}")
    print(f"   Storage: {results['storage_kb']:.1f}KB")
    
    # Save results
    os.makedirs(train_config.results_dir, exist_ok=True)
    result_file = os.path.join(
        train_config.results_dir,
        f"results_{method_config.name}_{model_config.name}.json"
    )
    # Convert any non-serializable values
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    save_results[k][k2] = {k3: float(v3) if isinstance(v3, (int, float)) else str(v3) for k3, v3 in v2.items()}
                else:
                    save_results[k][k2] = float(v2) if isinstance(v2, (int, float)) else str(v2)
        else:
            save_results[k] = float(v) if isinstance(v, (int, float)) else str(v)
    
    with open(result_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"   Results saved to {result_file}")
    
    # Clean up
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results
