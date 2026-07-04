"""
avr.cli — `avr train config.yaml` command-line entry point.

Usage:
    avr train configs/trace_lfm350m.yaml
    avr train configs/trace_qwen05b.yaml --seed 123

The config drives everything: model, LoRA, stream, learn, verify, repair.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import yaml
import torch
import numpy as np


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_model(config: dict):
    """Load HF model + attach LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    model_cfg = config["model"]
    model_id = model_cfg["id"]
    print(f"  Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    lora_cfg = LoraConfig(
        r=model_cfg.get("lora_rank", 32),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        target_modules=model_cfg["lora_targets"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    print(f"  LoRA attached (rank={model_cfg.get('lora_rank', 32)})")
    return model, tokenizer


def run_from_config(config_path: str, seed_override: int = None) -> dict:
    """Main entry: load config, build model, run stream, return results."""
    from .strategy import AVRStrategy
    from .data import load_stream
    from .metrics import compute_metrics, evaluate_task_accuracy
    from .framework import TaskSpec

    config = load_config(config_path)
    seed = seed_override or config.get("stream", {}).get("seed", 42)
    print(f"\n{'='*70}")
    print(f"avr-cl | seed={seed}")
    print(f"{'='*70}")

    # Seed everything
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Output dir
    output_dir = Path(config.get("output", {}).get("results_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stream
    stream_cfg = config["stream"]
    print(f"\nLoading benchmark: {stream_cfg['benchmark']}")
    stream_kwargs = {}
    if "n_train" in stream_cfg:
        stream_kwargs["n_train"] = stream_cfg["n_train"]
    if "n_eval" in stream_cfg:
        stream_kwargs["n_eval"] = stream_cfg["n_eval"]
    tasks = load_stream(
        benchmark=stream_cfg["benchmark"],
        output_dir=output_dir,
        tasks=stream_cfg.get("tasks"),
        seed=seed,
        **stream_kwargs,
    )
    T = len(tasks)

    # Build model
    print(f"\nBuilding model...")
    model, tokenizer = create_model(config)

    # R matrix
    R = [[0.0] * T for _ in range(T)]

    # Build the callback that fills R as tasks complete
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # We need test_pairs accessible from the callback; stash on a list
    test_pairs_per_task = [t.eval_pairs for t in tasks]

    def on_task_complete(m, tok, state, i):
        for j in range(i + 1):
            R[i][j] = evaluate_task_accuracy(
                m, tok, test_pairs_per_task[j], tasks[j].name,
                device=device,
            )

    # Run
    strategy = AVRStrategy(config)
    state = strategy.run(model, tokenizer, tasks, on_task_complete)

    # Metrics
    task_order = [t.name for t in tasks]
    metrics = compute_metrics(R, task_order)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  ACC: {metrics['ACC']:.3f}")
    print(f"  BWT: {metrics['BWT']:.3f}")
    print(f"  FF:  {metrics['FF']:.3f}")
    print(f"  Total repair steps: {state.total_repair_steps}")
    print(f"\n  R matrix:")
    header = "  After\\Test  " + "  ".join(f"{t[:8]:<10}" for t in task_order)
    print(header)
    for i in range(T):
        row = f"  {task_order[i][:8]:<10} " + "  ".join(
            f"{R[i][j]:<10.3f}" for j in range(T))
        print(row)

    # Save
    results = {
        "config": str(config_path),
        "seed": seed,
        "tasks": task_order,
        "metrics": metrics,
        "R": R,
        "total_repair_steps": state.total_repair_steps,
        "repair_log": state.repair_log,
        "best_ppls": state.best_ppls,
    }
    out_name = Path(config_path).stem
    if seed_override:
        out_name += f"_s{seed_override}"
    out_path = output_dir / f"{out_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        prog="avr",
        description="avr-cl: continual post-training framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train on a stream from a YAML config")
    train.add_argument("config", help="Path to YAML config file")
    train.add_argument("--seed", type=int, default=None,
                       help="Override seed from config")

    args = parser.parse_args()
    if args.command == "train":
        run_from_config(args.config, args.seed)


if __name__ == "__main__":
    main()
