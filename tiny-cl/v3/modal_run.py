"""
V3 Modal Runner: The Living Model on LFM2.5-350M with A100 GPU.
"""

import modal
import os

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "sentencepiece",
        "protobuf",
        "scipy",
    )
    .add_local_dir(
        local_path=os.path.join(os.path.dirname(__file__), "."),
        remote_path="/root/tiny-cl/v3",
    )
)

app = modal.App("tiny-cl-v3", image=image)


@app.function(gpu="A100", timeout=600)
def run_smoke_test():
    import subprocess
    result = subprocess.run(
        ["python", "/root/tiny-cl/v3/smoke_test.py"],
        capture_output=True, text=True, cwd="/root/tiny-cl/v3"
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode


@app.function(gpu="A100", timeout=14400)
def run_living():
    import sys
    sys.path.insert(0, "/root/tiny-cl/v3")
    from train import run_living_model
    return run_living_model(model_name="lfm2.5-350M", experiment="hero")


@app.function(gpu="A100", timeout=14400)
def run_naive():
    import sys
    sys.path.insert(0, "/root/tiny-cl/v3")
    from train import run_naive_baseline
    return run_naive_baseline(model_name="lfm2.5-350M", experiment="hero")


@app.function(gpu="A100", timeout=14400)
def run_both():
    import sys
    sys.path.insert(0, "/root/tiny-cl/v3")
    from train import run_living_model, run_naive_baseline

    print("\n" + "=" * 70)
    print("RUNNING NAIVE BASELINE FIRST")
    print("=" * 70)
    naive_results = run_naive_baseline(model_name="lfm2.5-350M", experiment="hero")

    print("\n" + "=" * 70)
    print("NOW RUNNING THE LIVING MODEL")
    print("=" * 70)
    living_results = run_living_model(model_name="lfm2.5-350M", experiment="hero")

    # Print comparison
    print("\n" + "#" * 70)
    print("# COMPARISON: LIVING MODEL vs NAIVE BASELINE")
    print("#" * 70)

    domains = ["A", "B", "C"]
    for pk in domains:
        naive_after = naive_results["phases"][pk].get("perplexity", {}).get(pk, 1)
        naive_final = naive_results["phases"][list(naive_results["phases"].keys())[-1]].get("perplexity", {}).get(pk, 0)
        living_after = living_results["phases"][pk].get("perplexity", {}).get(pk, 1)
        living_final = living_results["phases"][list(living_results["phases"].keys())[-1]].get("perplexity", {}).get(pk, 0)

        naive_ff = naive_final / naive_after if naive_after > 0 else float("inf")
        living_ff = living_final / living_after if living_after > 0 else float("inf")
        improvement = naive_ff / living_ff if living_ff > 0 else float("inf")
        print(f"  Phase {pk}: Naive={naive_ff:.2f}x, Living={living_ff:.2f}x ({improvement:.1f}x better)")

    return {"naive": naive_results, "living": living_results}


@app.local_entrypoint()
def main(target: str = "smoke"):
    """
    Run V3 Living Model experiments on Modal.

    Usage:
        modal run v3/modal_run.py --target smoke     # Pre-flight test
        modal run v3/modal_run.py --target naive      # Naive baseline only
        modal run v3/modal_run.py --target living     # Living Model only
        modal run v3/modal_run.py --target both       # Full comparison
    """
    if target == "smoke":
        print("Running smoke test...")
        result = run_smoke_test.remote()
        print(f"Smoke test exit code: {result}")
    elif target == "naive":
        print("Running naive baseline on LFM2.5-350M...")
        run_naive.remote()
        print("Done!")
    elif target == "living":
        print("Running THE LIVING MODEL on LFM2.5-350M...")
        run_living.remote()
        print("Done!")
    elif target == "both":
        print("Running full comparison on LFM2.5-350M...")
        run_both.remote()
        print("Done!")
