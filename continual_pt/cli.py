from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .loop import ContinualLearningLoop
from .schema import LearningGoal


def main() -> None:
    parser = argparse.ArgumentParser(description="Web-grounded autonomous continual post-training")
    subcommands = parser.add_subparsers(dest="command", required=True)
    learn = subcommands.add_parser("learn", help="Research and learn a real user-specified goal")
    learn.add_argument("goal", help="Goal YAML with held-out target and retention evaluation")
    learn.add_argument("--cycles", type=int, default=1)
    learn.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    learn.add_argument("--output", default="runs")
    learn.add_argument("--lora-rank", type=int, default=16)
    learn.add_argument("--repair-alpha", type=float, default=0.10)
    learn.add_argument("--max-repairs", type=int, default=10)
    args = parser.parse_args()

    if args.command == "learn":
        config_path = Path(args.goal)
        goal = LearningGoal.from_dict(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        run_dir = Path(args.output) / goal.name
        loop = ContinualLearningLoop(
            goal,
            run_dir,
            model_id=args.model,
            lora_rank=args.lora_rank,
            repair_alpha=args.repair_alpha,
            max_repairs=args.max_repairs,
        )
        result = loop.run(cycles=args.cycles)
        print(json.dumps(result["final"], indent=2))


if __name__ == "__main__":
    main()
