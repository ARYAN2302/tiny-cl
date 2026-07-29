"""Modal entry point. Authenticate separately, then run this on one T4."""
from __future__ import annotations

import modal

app = modal.App("continual-pt")
results_volume = modal.Volume.from_name("continual-pt-results", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("continual-pt-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=1.0",
        "beautifulsoup4>=4.12",
        "bitsandbytes>=0.43",
        "ddgs>=9.0",
        "peft>=0.13",
        "pyyaml>=6.0",
        "requests>=2.31",
        "torch>=2.2",
        "transformers>=4.45",
    )
    .add_local_dir("continual_pt", remote_path="/root/continual_pt")
)


@app.function(
    gpu="T4",
    timeout=60 * 60,
    image=image,
    volumes={"/results": results_volume, "/cache": hf_cache_volume},
)
def run_goal(goal_yaml: str, cycles: int = 1, model: str = "Qwen/Qwen3-4B-Instruct-2507") -> dict:
    import sys
    import os
    import time
    import yaml

    sys.path.insert(0, "/root")
    from continual_pt.loop import ContinualLearningLoop
    from continual_pt.schema import LearningGoal

    goal = LearningGoal.from_dict(yaml.safe_load(goal_yaml))
    run_name = f"{goal.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    output = f"/results/{run_name}"
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    print("[modal] loading model", flush=True)
    loop = ContinualLearningLoop(goal, output, model_id=model)
    # Persist the downloaded base model before long-running evaluation/training.
    hf_cache_volume.commit()
    print("[modal] model cache committed", flush=True)
    result = loop.run(cycles=cycles)
    results_volume.commit()
    result["artifact_path"] = output
    return result
