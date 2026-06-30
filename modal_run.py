"""
Modal wrapper for running Tiny-CL experiments on Modal.

Usage:
    1. pip install modal
    2. modal run modal_run.py
    
Or run specific experiments:
    modal run modal_run.py --methods naive anchor_cont --model-size 30M
"""

import modal
import sys

# Define the Modal app
app = modal.App("tiny-cl")

# Build image: CUDA + Python + dependencies + our code
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "wget")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "tqdm",
        "matplotlib",
    )
    # Copy our project code into the container at /root/tiny-cl
    .add_local_dir(
        local_path=".",
        remote_path="/root/tiny-cl",
    )
)

# Volume for caching downloaded datasets
volume = modal.Volume.from_name("tiny-cl-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,  # 2 hours — enough for all 5 methods + data download
    volumes={"/data": volume},
    memory=16384,
)
def run_experiments(
    model_size: str = "30M",
    methods: list = None,
    debug: bool = False,
    epochs: int = None,
):
    """Run experiments on Modal with A100 GPU."""
    import os
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = "/data/hf_cache"
    
    if methods is None:
        methods = ["naive", "freeze", "replay", "anchor_cont", "anchor_disc"]
    
    print(f"Running on Modal: model_size={model_size}, methods={methods}")
    print(f"Debug: {debug}, Epochs: {epochs}")
    
    sys.path.insert(0, "/root/tiny-cl")
    
    from config import MODEL_CONFIGS, METHOD_CONFIGS, TrainConfig
    from train import run_experiment
    
    train_config = TrainConfig(
        results_dir="/data/results",
        data_dir="/data/hf_cache",
        debug=debug,
    )
    if epochs:
        train_config.epochs_per_phase = epochs
    
    all_results = {}
    
    for method_name in methods:
        if method_name not in METHOD_CONFIGS:
            print(f"Unknown method: {method_name}, skipping")
            continue
        
        model_config = MODEL_CONFIGS[model_size]
        method_config = METHOD_CONFIGS[method_name]
        
        try:
            results = run_experiment(
                model_config=model_config,
                method_config=method_config,
                train_config=train_config,
                cache_dir="/data/hf_cache",
            )
            all_results[f"{method_name}_{model_size}"] = results
        except Exception as e:
            print(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    import json
    summary_path = "/data/results/summary.json"
    os.makedirs("/data/results", exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({k: str(v) for k, v in all_results.items()}, f, indent=2)
    
    volume.commit()
    print(f"\nResults saved to Modal volume: /data/results/")
    
    return all_results


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,
    volumes={"/data": volume},
    memory=16384,
)
def generate_plots():
    """Generate plots from saved results."""
    import os
    os.environ["HF_HOME"] = "/data/hf_cache"
    
    sys.path.insert(0, "/root/tiny-cl")
    from plot_results import generate_all_plots
    
    generate_all_plots("/data/results", ["30M"])
    
    volume.commit()
    print("Plots saved and volume committed.")


@app.local_entrypoint()
def main(
    model_size: str = "30M",
    methods: str = "",
    debug: bool = False,
    epochs: int = 0,
    plots_only: bool = False,
):
    """
    Entry point for: modal run modal_run.py
    
    Options:
        --model-size 30M or 15M
        --methods "naive anchor_cont anchor_disc" (space-separated)
        --debug  Use tiny data subset for testing
        --epochs N  Override epochs per phase
        --plots-only  Just generate plots from existing results
    """
    method_list = methods.split() if methods else None
    epochs_val = epochs if epochs > 0 else None
    
    if plots_only:
        generate_plots.remote()
    else:
        results = run_experiments.remote(
            model_size=model_size,
            methods=method_list,
            debug=debug,
            epochs=epochs_val,
        )
        print(f"\nDone! {len(results)} experiments completed.")
