"""
V4 Training Loop: Streaming continual learning with variable increment sizes.
Runs on M1 Mac — no GPU required.
"""

import os
import json
import time
import random
import torch
from torch.utils.data import DataLoader

from config import (
    MODEL_CONFIGS, DOMAINS, INCREMENT_SIZES, METHODS,
    TrainConfig, AVRConfig, EWCConfig,
)
from models import create_model
from data import prepare_all_domains, StreamingDataset, TokenDataset
from methods import create_method
from evaluate import evaluate_all_phases, compute_perplexity


def run_streaming_experiment(
    model_name: str,
    method_name: str,
    increment_size: int = 0,  # 0 = full phase
    train_config: TrainConfig = None,
    seed: int = 42,
):
    if train_config is None:
        train_config = TrainConfig()

    torch.manual_seed(seed)
    random.seed(seed)

    device = train_config.device
    model_config = MODEL_CONFIGS[model_name]
    context_length = model_config.context_length

    print(f"\n{'#'*70}")
    print(f"# Streaming Experiment")
    print(f"# Model: {model_name} | Method: {method_name}")
    print(f"# Increment: {'full-phase' if increment_size == 0 else f'{increment_size} examples'}")
    print(f"# Device: {device}")
    print(f"{'#'*70}")

    # Create model
    model = create_model(model_name, device)

    # Prepare data
    phases_data, tokenizer = prepare_all_domains(
        DOMAINS,
        vocab_size=model_config.vocab_size,
        context_length=context_length,
        max_tokens=500_000,
        seed=seed,
    )

    # Create method
    avr_config = AVRConfig()
    ewc_config = EWCConfig()
    method = create_method(method_name, train_config, avr_config, ewc_config)

    # Create validation datasets
    val_datasets = {}
    for pk, data in phases_data.items():
        val_datasets[pk] = TokenDataset(data["val_tokens"], context_length)

    # Create streaming datasets
    streaming_datasets = {}
    for pk, data in phases_data.items():
        streaming_datasets[pk] = StreamingDataset(
            data["train_tokens"], context_length, increment_size
        )

    all_results = {
        "model": model_name,
        "method": method_name,
        "increment_size": increment_size,
        "phases": {},
    }

    for phase_key in DOMAINS.keys():
        sdata = streaming_datasets[phase_key]
        method.on_phase_start(model, phase_key)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=train_config.lr,
                                      weight_decay=train_config.weight_decay)

        global_step = 0
        total_loss = 0.0
        phase_start = time.time()

        print(f"\n{'='*50}")
        print(f"Phase {phase_key}: {DOMAINS[phase_key].display_name}")
        print(f"  {sdata.n_increments} increments × {sdata.increment_size} examples")
        print(f"{'='*50}")

        for inc_idx in range(sdata.n_increments):
            batch = sdata.get_increment_tensor(inc_idx, device)
            model.train()

            loss, metrics = method.compute_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, train_config.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # AVR: check after each increment
            if hasattr(method, 'on_increment_end'):
                method.on_increment_end(model, phase_key, device)

            if global_step % 50 == 0:
                print(f"  Inc {inc_idx+1}/{sdata.n_increments} | Loss: {loss.item():.4f}")

        method.on_phase_end(model, phase_key, sdata, device)

        # Evaluate
        eval_results = evaluate_all_phases(model, val_datasets, 
                                          method.completed_phases, device,
                                          train_config.eval_samples)
        eval_results["avg_loss"] = total_loss / max(global_step, 1)
        eval_results["extra_steps"] = method.extra_steps if hasattr(method, 'extra_steps') else 0
        if hasattr(method, 'total_repairs'):
            eval_results["total_repairs"] = method.total_repairs
        if hasattr(method, 'total_verifies'):
            eval_results["total_verifies"] = method.total_verifies
        if hasattr(method, 'ewc_computable'):
            eval_results["ewc_computable"] = method.ewc_computable

        all_results["phases"][phase_key] = eval_results

        print(f"\n  Phase {phase_key} results:")
        for pk, ppl in eval_results.get("perplexity", {}).items():
            print(f"    {pk}: PPL={ppl:.2f}")

        phase_time = time.time() - phase_start
        eval_results["training_time"] = phase_time

    # Save results
    os.makedirs(train_config.results_dir, exist_ok=True)
    inc_str = "full" if increment_size == 0 else str(increment_size)
    result_file = os.path.join(
        train_config.results_dir,
        f"{model_name}_{method_name}_inc{inc_str}.json"
    )
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return all_results


def run_full_grid():
    """Run the complete experiment grid."""
    all_grid_results = []

    for model_name in MODEL_CONFIGS.keys():
        for inc_size in INCREMENT_SIZES:
            for method_name in METHODS:
                print(f"\n{'*'*70}")
                print(f"* GRID: {model_name} | {method_name} | inc={inc_size}")
                print(f"{'*'*70}")
                try:
                    result = run_streaming_experiment(
                        model_name=model_name,
                        method_name=method_name,
                        increment_size=inc_size,
                    )
                    all_grid_results.append(result)
                except Exception as e:
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Save combined results
    os.makedirs("v4/results", exist_ok=True)
    with open("v4/results/full_grid.json", "w") as f:
        json.dump(all_grid_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"GRID COMPLETE: {len(all_grid_results)} runs")
    print(f"{'='*70}")

    return all_grid_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm_1.4M", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", default="avr", choices=METHODS)
    parser.add_argument("--increment", default=0, type=int, help="0=full, else examples per increment")
    parser.add_argument("--grid", action="store_true", help="Run full grid")
    args = parser.parse_args()

    if args.grid:
        run_full_grid()
    else:
        run_streaming_experiment(args.model, args.method, args.increment)
