"""
avr.data — task stream loaders.

TRACE: 8-task CL benchmark (we use 4: C-STANCE, FOMC, NumGLUE-cm, NumGLUE-ds).
MMLU stream: pick N subjects from MMLU, treat as a CL stream.

Both produce List[TaskSpec] for the framework.
"""

from __future__ import annotations
from typing import List, Optional
import json
import os
from pathlib import Path
from .framework import TaskSpec


# ────────────────────────────────────────────────────────────────────
# TRACE
# ────────────────────────────────────────────────────────────────────

TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"
TRACE_VARIANT = "LLM-CL-Benchmark_5000"
TRACE_TASKS_ALL = ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds",
                   "MeetingBank", "ScienceQA", "CausalInference", "contractNLI"]
TRACE_TASKS_DEFAULT = ["C-STANCE", "FOMC", "NumGLUE-cm", "NumGLUE-ds"]


def download_trace(output_dir: Path) -> Path:
    """Download TRACE from Google Drive if not present."""
    trace_dir = output_dir / "trace_data"
    if trace_dir.exists() and any(trace_dir.iterdir()):
        return trace_dir
    print("  Downloading TRACE...")
    import gdown
    import zipfile
    zip_path = output_dir / "trace_benchmark.zip"
    gdown.download(id=TRACE_GDRIVE_ID, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(output_dir)
    for d in output_dir.rglob(f"*{TRACE_VARIANT}*"):
        if d.is_dir():
            trace_dir = d
            break
    print(f"  TRACE at: {trace_dir}")
    return trace_dir


def load_trace_task(trace_dir: Path, task_name: str) -> TaskSpec:
    task_dir = trace_dir / task_name
    with open(task_dir / "train.json") as f:
        train_data = json.load(f)
    with open(task_dir / "test.json") as f:
        test_data = json.load(f)
    train_pairs = [(ex["prompt"], ex["answer"]) for ex in train_data]
    test_pairs = [(ex["prompt"], ex["answer"]) for ex in test_data]
    print(f"    {task_name}: {len(train_pairs)} train, {len(test_pairs)} test")
    return TaskSpec(
        name=task_name,
        train_pairs=train_pairs,
        eval_pairs=test_pairs,
        metadata={"benchmark": "trace"},
    )


def load_trace(output_dir: Path,
               tasks: Optional[List[str]] = None) -> List[TaskSpec]:
    """Load TRACE tasks as a stream of TaskSpec."""
    trace_dir = download_trace(output_dir)
    task_names = tasks or TRACE_TASKS_DEFAULT
    return [load_trace_task(trace_dir, t) for t in task_names]


# ────────────────────────────────────────────────────────────────────
# MMLU stream
# ────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS_DEFAULT = [
    "abstract_algebra", "college_physics", "jurisprudence",
    "global_facts", "anatomy", "business_ethics",
    "clinical_knowledge", "college_chemistry",
]


def load_mmlu_stream(output_dir: Path,
                     subjects: Optional[List[str]] = None,
                     n_train: int = 500,
                     n_eval: int = 200) -> List[TaskSpec]:
    """Load MMLU subjects as a CL stream. Uses cais/mmlu HF dataset."""
    from datasets import load_dataset
    subjects = subjects or MMLU_SUBJECTS_DEFAULT
    stream = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        # Format: question, choices (4), answer (0-3)
        pairs = []
        for ex in ds:
            q = ex["question"]
            choices = ex["choices"]
            ans = ex["answer"]
            # MCQ format: prompt with A/B/C/D, gold = letter
            prompt = f"{q}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
            gold = ["A", "B", "C", "D"][ans]
            pairs.append((prompt, gold))
        # Split into train/eval
        random.shuffle(pairs)  # not seeded here; set seed in caller
        train_pairs = pairs[:n_train]
        eval_pairs = pairs[n_train:n_train + n_eval]
        print(f"    MMLU {subj}: {len(train_pairs)} train, {len(eval_pairs)} eval")
        stream.append(TaskSpec(
            name=f"mmlu_{subj}",
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            metadata={"benchmark": "mmlu", "subject": subj},
        ))
    return stream


# Need this import here for the shuffle
import random


# ────────────────────────────────────────────────────────────────────
# Real-world domain stream
# ────────────────────────────────────────────────────────────────────

REALWORLD_DOMAINS = {
    "medical": {
        "dataset": "epfl-llm/guidelines",
        "field": "clean_text",
        "display": "Medical",
    },
    "code": {
        "dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "field": "output",
        "display": "Code",
    },
    "creative": {
        "dataset": "roneneldan/TinyStories",
        "field": "text",
        "display": "Creative",
    },
    "legal": {
        "dataset": "nguha/legalbench",
        "field": "text",
        "display": "Legal",
    },
    "finance": {
        "dataset": "gbharti/finance-alpaca",
        "field": "output",
        "display": "Finance",
    },
}


def load_realworld_stream(output_dir: Path,
                          tasks: Optional[List[str]] = None,
                          n_train: int = 1000,
                          n_eval: int = 200) -> List[TaskSpec]:
    """Load real-world domain stream: Medical, Code, Creative, Legal, Finance.

    Each domain is a different HF dataset, treated as one task in the stream.
    Tests AVR on genuinely different domains (not just MMLU subjects).
    """
    from datasets import load_dataset
    domains = tasks or list(REALWORLD_DOMAINS.keys())
    stream = []
    for domain_key in domains:
        d = REALWORLD_DOMAINS[domain_key]
        print(f"    Loading {d['display']}...")
        try:
            ds = load_dataset(d["dataset"], split="train")
            texts = [t for t in ds[d["field"]] if t and len(t.strip()) > 10]
            random.shuffle(texts)
            # Build (prompt, answer) pairs — for generative domains, we split
            # text into prompt (first half) + answer (second half)
            pairs = []
            for text in texts[:n_train + n_eval]:
                mid = len(text) // 2
                prompt = text[:mid].strip()
                answer = text[mid:].strip()
                if prompt and answer:
                    pairs.append((prompt, answer))
            train_pairs = pairs[:n_train]
            eval_pairs = pairs[n_train:n_train + n_eval]
            print(f"    {d['display']}: {len(train_pairs)} train, {len(eval_pairs)} eval")
            stream.append(TaskSpec(
                name=domain_key,
                train_pairs=train_pairs,
                eval_pairs=eval_pairs,
                metadata={"benchmark": "realworld", "domain": domain_key},
            ))
        except Exception as e:
            print(f"    WARNING: {d['display']} failed to load: {e}")
            print(f"    Skipping {domain_key}")
    return stream


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────

def load_stream(benchmark: str, output_dir: Path,
                tasks: Optional[List[str]] = None,
                seed: int = 42, **kwargs) -> List[TaskSpec]:
    """Dispatch to the right loader based on benchmark name."""
    random.seed(seed)
    if benchmark == "trace":
        return load_trace(output_dir, tasks)
    elif benchmark == "mmlu_stream":
        return load_mmlu_stream(output_dir, tasks, **kwargs)
    elif benchmark == "realworld_stream":
        return load_realworld_stream(output_dir, tasks, **kwargs)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. "
                         f"Available: trace, mmlu_stream, realworld_stream")
