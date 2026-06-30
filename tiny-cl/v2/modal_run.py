"""
V2 Modal Runner: Cloud execution on Modal with A100 GPU.
OPTIMIZED: loads model + data once, runs all methods sequentially.
"""

import modal
import os

# ──────────────────────────────────────────────
# Modal Image Setup
# ──────────────────────────────────────────────

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
        remote_path="/root/tiny-cl/v2",
    )
)

app = modal.App("tiny-cl-v2", image=image)


# ──────────────────────────────────────────────
# Smoke Test
# ──────────────────────────────────────────────

@app.function(gpu="A100", timeout=600)
def run_smoke_test():
    """Run smoke test to verify models work on Modal."""
    import subprocess
    result = subprocess.run(
        ["python", "/root/tiny-cl/v2/smoke_test.py"],
        capture_output=True, text=True, cwd="/root/tiny-cl/v2"
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode


# ──────────────────────────────────────────────
# Optimized Batch Runner — loads model + data ONCE
# ──────────────────────────────────────────────

@app.function(gpu="A100", timeout=14400)
def run_batch_optimized(model_name: str, methods: list, experiment: str):
    """
    Run multiple methods on the same model + data.
    Loads model and prepares data ONCE, then runs all methods sequentially.
    """
    import sys
    sys.path.insert(0, "/root/tiny-cl/v2")

    import json
    import torch
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    from config import (
        PRETRAINED_MODELS, METHOD_CONFIGS, EXPERIMENT_CONFIGS,
        TrainConfig, LoRAConfig,
    )
    from data import prepare_all_domains, create_dataloader
    from methods import create_method, AnchorAVRDiscrete
    from evaluate import evaluate_all_phases

    model_config = PRETRAINED_MODELS[model_name]
    domains = EXPERIMENT_CONFIGS[experiment]
    train_config = TrainConfig(
        device="cuda",
        results_dir="/root/tiny-cl/v2/results",
        data_dir="/root/tiny-cl/v2/data",
    )
    lora_config = LoRAConfig()
    seed = 42

    # ─── LOAD MODEL + DATA ONCE ───
    print(f"\n{'#'*70}")
    print(f"# Loading {model_config.display_name} + data (ONCE for all methods)")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    phases_data, tokenizer = prepare_all_domains(
        domains, model_config.hf_id, train_config, seed=seed
    )

    # Validation datasets
    val_datasets = {}
    for pk, pd in phases_data.items():
        val_datasets[pk] = pd["val"]

    all_results = []

    # ─── RUN EACH METHOD ───
    for method_name in methods:
        torch.manual_seed(seed)
        random.seed(seed)

        method_config = METHOD_CONFIGS[method_name]

        print(f"\n{'='*60}")
        print(f"Method: {method_config.display_name}")
        print(f"{'='*60}")

        # Fresh model + LoRA for each method
        model = AutoModelForCausalLM.from_pretrained(
            model_config.hf_id,
            dtype=torch.float32,
            trust_remote_code=model_config.trust_remote_code,
        )
        model = model.to("cuda")

        target_modules = lora_config.target_modules or model_config.lora_target_modules
        peft_config = LoraConfig(
            r=lora_config.rank, lora_alpha=lora_config.alpha,
            lora_dropout=lora_config.dropout, target_modules=target_modules,
            bias=lora_config.bias, task_type=lora_config.task_type,
        )
        model = get_peft_model(model, peft_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable: {trainable:,} / {total:,}")

        # Create method
        method = create_method(method_name, method_config, train_config)

        # Train sequentially on each phase
        method_results = {
            "model": model_name,
            "method": method_name,
            "experiment": experiment,
            "phases": {},
        }

        for phase_key in domains.keys():
            train_dataset = phases_data[phase_key]["train"]
            train_loader = create_dataloader(train_dataset, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
            trainable_params = [p for p in model.parameters() if p.requires_grad]

            method.on_phase_start(model, phase_key, val_datasets)

            optimizer = torch.optim.AdamW(trainable_params, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
            n_steps = len(train_loader) * train_config.epochs_per_phase
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config.warmup_steps, num_training_steps=n_steps)

            global_step = 0
            total_loss = 0.0

            for epoch in range(train_config.epochs_per_phase):
                for batch in train_loader:
                    model.train()
                    loss, metrics = method.compute_loss(model, batch, "cuda")

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()
                    global_step += 1

                    if isinstance(method, AnchorAVRDiscrete) and method.should_verify():
                        method.verify_and_repair(model, "cuda")

                    if global_step % train_config.log_interval == 0:
                        print(f"  [{method_name}] Phase {phase_key} Step {global_step}/{n_steps} | Loss: {loss.item():.4f}")

            method.on_phase_end(model, phase_key, train_dataset)

            # Evaluate
            eval_results = evaluate_all_phases(model, val_datasets, method.completed_phases, "cuda", train_config.eval_samples)
            eval_results["avg_loss"] = total_loss / max(global_step, 1)
            if isinstance(method, AnchorAVRDiscrete):
                eval_results["total_repairs"] = method.total_repairs
            method_results["phases"][phase_key] = eval_results

        method_results["storage_kb"] = method.get_storage_kb()

        # Print final summary
        last_phase_key = list(domains.keys())[-1]
        final_ppls = method_results["phases"][last_phase_key].get("perplexity", {})
        print(f"\n  FINAL: {method_config.display_name}")
        for pk, ppl in final_ppls.items():
            print(f"    Phase {pk}: {ppl:.2f}")

        # Forgetting factor
        for pk in domains.keys():
            after_learn = method_results["phases"][pk].get("perplexity", {}).get(pk, None)
            after_all = final_ppls.get(pk, None)
            if after_learn and after_all and after_learn > 0:
                print(f"    Phase {pk} forgetting: {after_all/after_learn:.2f}x")

        all_results.append(method_results)

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Save combined results
    os.makedirs(train_config.results_dir, exist_ok=True)
    result_file = os.path.join(train_config.results_dir, f"combined_{model_name}_{experiment}.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    return all_results


# ──────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────

@app.function(gpu="A100", timeout=14400)
def run_hero():
    return run_batch_optimized.remote(
        model_name="smollm2-360M",
        methods=["naive", "ewc", "anchor_disc"],
        experiment="hero",
    )

@app.function(gpu="A100", timeout=14400)
def run_lfm_moonshot():
    return run_batch_optimized.remote(
        model_name="lfm2.5-350M",
        methods=["naive", "anchor_disc"],
        experiment="hero",
    )

@app.function(gpu="A100", timeout=14400)
def run_ablation():
    return run_batch_optimized.remote(
        model_name="smollm2-135M",
        methods=["naive", "anchor_disc"],
        experiment="hero",
    )

@app.function(gpu="A100", timeout=14400)
def run_sanity():
    return run_batch_optimized.remote(
        model_name="smollm2-360M",
        methods=["naive", "anchor_disc"],
        experiment="sanity",
    )

@app.function(gpu="A100", timeout=14400)
def run_anchor_only():
    """Run ONLY Anchor-AVR Discrete (after Naive + EWC already done)."""
    return run_batch_optimized.remote(
        model_name="smollm2-360M",
        methods=["anchor_disc"],
        experiment="hero",
    )

@app.function(gpu="A100", timeout=14400)
def run_baselines_1epoch():
    """Run Naive + EWC at 1 epoch for fair comparison with Anchor-AVR."""
    return run_batch_optimized.remote(
        model_name="smollm2-360M",
        methods=["naive", "ewc"],
        experiment="hero",
    )


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

@app.local_entrypoint()
def main(
    target: str = "hero",
):
    """
    Run V2 experiments on Modal.

    Usage:
        modal run v2/modal_run.py --target hero
        modal run v2/modal_run.py --target lfm
        modal run v2/modal_run.py --target ablation
        modal run v2/modal_run.py --target sanity
        modal run v2/modal_run.py --target smoke
    """
    if target == "smoke":
        print("Running smoke test...")
        result = run_smoke_test.remote()
        print(f"Smoke test exit code: {result}")
    elif target == "sanity":
        print("Running sanity...")
        results = run_sanity.remote()
        print(f"Done! {len(results)} experiments")
    elif target == "hero":
        print("Running hero (SmolLM2-360M, 3 methods)...")
        results = run_hero.remote()
        print(f"Done! {len(results)} experiments")
    elif target == "anchor":
        print("Running Anchor-AVR Discrete only...")
        results = run_anchor_only.remote()
        print(f"Done! {len(results)} experiments")
    elif target == "baselines":
        print("Running Naive + EWC at 1 epoch (fair comparison)...")
        results = run_baselines_1epoch.remote()
        print(f"Done! {len(results)} experiments")
    elif target == "lfm":
        print("Running LFM2.5 moon shot...")
        results = run_lfm_moonshot.remote()
        print(f"Done! {len(results)} experiments")
    elif target == "ablation":
        print("Running 135M ablation...")
        results = run_ablation.remote()
        print(f"Done! {len(results)} experiments")
