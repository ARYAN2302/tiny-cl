"""
avr.cli — `avr train config.yaml`

Config format:
    model: Qwen/Qwen3-1.7B
    lora_rank: 128
    tasks:
      - name: task_a
        train: data/task_a_train.json
        eval: data/task_a_eval.json
      - name: task_b
        train: data/task_b_train.json
        eval: data/task_b_eval.json

Data format (JSON): list of [question, answer, gold] triples.
"""
import argparse, json, sys
from pathlib import Path


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_task_data(task_cfg):
    with open(task_cfg["train"]) as f:
        train = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]
    with open(task_cfg["eval"]) as f:
        eval_data = [(ex[0], ex[1], ex[2]) for ex in json.load(f)]
    return train, eval_data


def main():
    parser = argparse.ArgumentParser(
        prog="avr",
        description="avr-cl: detect when fine-tuning broke old tasks and repair them")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run on a task stream")
    train.add_argument("config", help="Path to YAML config")
    train.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "train":
        import avr
        config = load_config(args.config)
        tasks = []
        for task_cfg in config["tasks"]:
            name = task_cfg["name"]
            train_data, eval_data = load_task_data(task_cfg)
            tasks.append((name, train_data, eval_data))
            print(f"  {name}: {len(train_data)} train, {len(eval_data)} eval")

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
            seed=args.seed,
        )

        output = config.get("output", "results.json")
        with open(output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved: {output}")


if __name__ == "__main__":
    main()
