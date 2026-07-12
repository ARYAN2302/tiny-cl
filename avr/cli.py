"""
avr.cli — `avr train config.yaml` command-line entry point.
"""
import argparse, json, sys
from pathlib import Path


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_task_data(task_cfg):
    """Load train/eval data from JSON files. Each file is a list of
    [question, answer, gold] triples."""
    train_path = task_cfg["train"]
    eval_path = task_cfg["eval"]
    with open(train_path) as f:
        train = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]
    with open(eval_path) as f:
        eval_data = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]
    return train, eval_data


def main():
    parser = argparse.ArgumentParser(
        prog="avr",
        description="avr-cl: continual post-training with drift detection + repair")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run AVR on a task stream")
    train.add_argument("config", help="Path to YAML config")
    train.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    if args.command == "train":
        run_from_config(args.config, args.seed)


def run_from_config(config_path, seed_override=None):
    import avr
    config = load_config(config_path)
    seed = seed_override or config.get("seed", 42)

    # Load tasks
    tasks = []
    for task_cfg in config["tasks"]:
        name = task_cfg["name"]
        train_data, eval_data = load_task_data(task_cfg)
        tasks.append((name, train_data, eval_data))
        print(f"  {name}: {len(train_data)} train, {len(eval_data)} eval", flush=True)

    # Run
    result = avr.run(
        model=config["model"],
        tasks=tasks,
        lora_rank=config.get("lora_rank", 128),
        lora_alpha=config.get("lora_alpha", 128),
        lora_targets=config.get("lora_targets"),
        epochs=config.get("epochs", 3),
        lr=config.get("lr", 2e-4),
        batch_size=config.get("batch_size", 4),
        grad_accum=config.get("grad_accum", 4),
        ctx_len=config.get("ctx_len", 512),
        drift_threshold=config.get("drift_threshold", 1.15),
        repair_alpha=config.get("repair_alpha", 0.1),
        max_repair_steps=config.get("max_repair_steps", 10),
        two_stream=config.get("two_stream", False),
        seed=seed,
    )

    # Save results
    output = config.get("output", "results.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved: {output}", flush=True)


if __name__ == "__main__":
    main()
