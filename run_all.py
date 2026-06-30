"""
Run all experiments and collect results.
"""

import os
import sys
import json
import argparse
import time
import torch

from config import (
    MODEL_CONFIGS, METHOD_CONFIGS,
    REQUIRED_EXPERIMENTS, OPTIONAL_EXPERIMENTS,
    TrainConfig,
)
from train import run_experiment
from evaluate import format_results_table


def parse_args():
    parser = argparse.ArgumentParser(description="Run Tiny-CL experiments")
    parser.add_argument(
        "--model-size", type=str, default=None,
        help="Model size to use (30M or 15M). Default: run all required."
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Methods to run (naive freeze replay anchor_cont anchor_disc). Default: run all required."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments including optional ones"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: tiny data subset, fewer epochs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--data-cache", type=str, default="data",
        help="Directory to cache downloaded datasets"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs per phase"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (cuda/cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Training config
    train_config = TrainConfig(
        results_dir=args.output_dir,
        data_dir=args.data_cache,
        debug=args.debug,
    )
    if args.epochs:
        train_config.epochs_per_phase = args.epochs
    if args.batch_size:
        train_config.batch_size = args.batch_size
    if args.lr:
        train_config.learning_rate = args.lr
    if args.device:
        train_config.device = args.device
    
    # Determine which experiments to run
    if args.model_size and args.methods:
        experiments = [(args.model_size, m) for m in args.methods]
    elif args.model_size:
        experiments = [(args.model_size, m) for _, m in REQUIRED_EXPERIMENTS]
    elif args.methods:
        experiments = [("30M", m) for m in args.methods]
    else:
        experiments = list(REQUIRED_EXPERIMENTS)
        if args.all:
            experiments.extend(OPTIONAL_EXPERIMENTS)
    
    print(f"\n{'='*60}")
    print(f"Tiny-CL Experiment Runner")
    print(f"{'='*60}")
    print(f"Device: {train_config.device}")
    print(f"Debug: {train_config.debug}")
    print(f"Epochs per phase: {train_config.epochs_per_phase}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Learning rate: {train_config.learning_rate}")
    print(f"\nExperiments to run ({len(experiments)}):")
    for i, (model_size, method_name) in enumerate(experiments, 1):
        display = METHOD_CONFIGS[method_name].display_name
        print(f"  {i}. {display} ({model_size})")
    print(f"{'='*60}")
    
    # Create output directories
    os.makedirs(train_config.results_dir, exist_ok=True)
    os.makedirs(train_config.data_dir, exist_ok=True)
    
    # Run experiments
    all_results = {}
    total_start = time.time()
    
    for i, (model_size, method_name) in enumerate(experiments, 1):
        print(f"\n\n{'='*60}")
        print(f"Experiment {i}/{len(experiments)}")
        print(f"{'='*60}")
        
        model_config = MODEL_CONFIGS[model_size]
        method_config = METHOD_CONFIGS[method_name]
        
        try:
            results = run_experiment(
                model_config=model_config,
                method_config=method_config,
                train_config=train_config,
                cache_dir=train_config.data_dir,
            )
            all_results[f"{method_name}_{model_size}"] = results
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"{method_name}_{model_size}"] = {"error": str(e)}
    
    # Summary
    total_elapsed = time.time() - total_start
    print(f"\n\n{'#'*70}")
    print(f"# ALL EXPERIMENTS COMPLETE")
    print(f"# Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'#'*70}\n")
    
    # Print results table
    table = format_results_table(all_results)
    print(table)
    
    # Save summary
    summary_file = os.path.join(train_config.results_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "experiments": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in all_results.items()},
            "total_time_minutes": total_elapsed / 60,
            "config": {
                "epochs_per_phase": train_config.epochs_per_phase,
                "batch_size": train_config.batch_size,
                "learning_rate": train_config.learning_rate,
                "debug": train_config.debug,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {train_config.results_dir}/")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
