"""The autonomous web-to-weights continual-learning loop."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .evaluate import evaluate
from .learn import GroundedExampleBuilder, example_records, train_fast_weights
from .model import DEFAULT_MODEL, load_learner
from .research import WebResearcher, source_records
from .research_agent import ResearchAgent
from .retention import AVRRetentionGate, get_lora_state
from .schema import LearningGoal


class ContinualLearningLoop:
    def __init__(
        self,
        goal: LearningGoal,
        output_dir: str | Path,
        model_id: str = DEFAULT_MODEL,
        lora_rank: int = 16,
        repair_alpha: float = 0.10,
        max_repairs: int = 10,
    ):
        self.goal = goal
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.tokenizer = load_learner(model_id, lora_rank=lora_rank)
        self.researcher = WebResearcher()
        self.research_agent = ResearchAgent()
        self.example_builder = GroundedExampleBuilder()
        self.avr = AVRRetentionGate(repair_alpha=repair_alpha, max_repairs=max_repairs)

    def run(self, cycles: int = 1) -> dict:
        print(f"[loop] baseline evaluation for {self.goal.name}", flush=True)
        baseline_target = evaluate(self.model, self.tokenizer, self.goal.target_eval, "target-baseline")
        retention_baseline = self.avr.baseline(self.model, self.tokenizer, self.goal.retention_eval)
        queries = self.research_agent.plan_queries(self.model, self.tokenizer, self.goal)
        documents = self.researcher.collect(self.goal, queries=queries)
        documents = self.research_agent.select_sources(self.model, self.tokenizer, self.goal, documents)
        run = {
            "goal": asdict(self.goal),
            "baseline": {"target": baseline_target.to_dict(), "retention": retention_baseline.to_dict()},
            "sources": source_records(documents),
            "cycles": [],
        }
        self._write_json("run.json", run)

        current_target = baseline_target
        for cycle in range(1, cycles + 1):
            print(f"[loop] learning cycle {cycle}/{cycles}", flush=True)
            examples = self.example_builder.build(
                self.model, self.tokenizer, self.goal.objective, documents, max_examples=24
            )
            # Candidate self-edits differ in update budget, not in target data.
            # Each begins from the same anchor and the held-out outcome decides
            # which (if any) becomes durable.  The prior run only took three
            # optimizer steps, too little to test whether the curriculum could
            # be incorporated at all.
            schedules = [
                {"name": "balanced-all-lora", "epochs": 3, "lr": 5e-5, "target_modules": None},
                {"name": "balanced-all-lora-strong", "epochs": 5, "lr": 1e-4, "target_modules": None},
                {"name": "balanced-attention", "epochs": 5, "lr": 1e-4, "target_modules": ("q_proj", "v_proj", "o_proj")},
            ]
            candidates = []
            accepted = False
            for schedule in schedules:
                print(f"[loop] candidate update: {schedule['name']}", flush=True)
                anchor = get_lora_state(self.model)
                training = train_fast_weights(
                    self.model,
                    self.tokenizer,
                    examples,
                    epochs=schedule["epochs"],
                    lr=schedule["lr"],
                    grad_accum=4,
                    target_modules=schedule["target_modules"],
                )
                print("[loop] AVR target/retention gate", flush=True)
                accepted, avr_log = self.avr.commit_or_rollback(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    anchor=anchor,
                    retention_baseline=retention_baseline,
                    retained_cases=self.goal.retention_eval,
                    target_cases=self.goal.target_eval,
                    target_baseline=current_target,
                )
                candidates.append({"schedule": schedule["name"], "training": training, "avr": avr_log})
                if accepted:
                    break
            if accepted:
                current_target = evaluate(self.model, self.tokenizer, self.goal.target_eval, f"target-committed-{cycle}")
                retention_baseline = self.avr.baseline(self.model, self.tokenizer, self.goal.retention_eval)
                self.model.save_pretrained(self.output_dir / f"adapter-cycle-{cycle}")
            record = {
                "cycle": cycle,
                "grounded_examples": example_records(examples),
                "candidates": candidates,
            }
            run["cycles"].append(record)
            self._write_json("run.json", run)

        run["final"] = {
            "target": evaluate(self.model, self.tokenizer, self.goal.target_eval, "target-final").to_dict(),
            "retention": evaluate(self.model, self.tokenizer, self.goal.retention_eval, "retention-final").to_dict(),
        }
        self._write_json("run.json", run)
        print("[loop] completed", flush=True)
        return run

    def _write_json(self, name: str, payload: dict) -> None:
        (self.output_dir / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
