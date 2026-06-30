"""
V2 Experiment Runner: CLI entry point for running experiments.
"""

import argparse
import json
import os
import sys

from config import (
    PRETRAINED_MODELS, METHOD_CONFIGS, EXPERIMENT_CONFIGS,
    TrainConfig, LoRAConfig,
    SANITY_EXPERIMENTS, HERO_EXPERIMENTS, LFM_EXPERIMENTS, ABLATION_EXPERIMENTS,
)
from train import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Tiny-CL V2: The Living Model")
    parser.add_argument("--model", type=str, default="smollm2-360M",
                        choices=list(PRETRAINED_MODELS.keys()),
                        help="Pretrained model to use")
    parser.add_argument("--method", type=str, default="anchor_disc",
                        choices=list(METHOD_CONFIGS.keys()),
                        help="CL method to use")
    parser.add_argument("--experiment", type=str, default="hero",
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Experiment config (sanity or hero)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["sanity", "hero", "lfm", "ablation"],
                        help="Run a preset batch of experiments")
    parser.add_argument("--debug", action="store_true",
                        help="Use tiny data subset for debugging")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs per phase")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    train_config = TrainConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs_per_phase=args.epochs,
        debug=args.debug,
        seed=args.seed,
    )

    if args.preset:
        # Run a preset batch of experiments
        presets = {
            "sanity": SANITY_EXPERIMENTS,
            "hero": HERO_EXPERIMENTS,
            "lfm": LFM_EXPERIMENTS,
            "ablation": ABLATION_EXPERIMENTS,
        }
        experiments = presets[args.preset]
        print(f"Running preset: {args.preset} ({len(experiments)} experiments)")

        all_results = []
        for model_name, method_name, exp_type in experiments:
            print(f"\n{'*'*70}")
            print(f"* Next: {model_name} + {method_name} ({exp_type})")
            print(f"{'*'*70}")

            result = run_experiment(
                model_name=model_name,
                method_name=method_name,
                experiment_type=exp_type,
                train_config=TrainConfig(debug=args.debug, seed=args.seed),
                seed=args.seed,
            )
            all_results.append(result)

        # Save combined results
        os.makedirs(train_config.results_dir, exist_ok=True)
        combined_file = os.path.join(
            train_config.results_dir,
            f"combined_{args.preset}.json"
        )
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {combined_file}")

    else:
        # Run single experiment
        run_experiment(
            model_name=args.model,
            method_name=args.method,
            experiment_type=args.experiment,
            train_config=train_config,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
