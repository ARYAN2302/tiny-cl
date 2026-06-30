"""
V2 Training Loop: LoRA fine-tuning on pretrained models with CL methods.
Handles sequential phase training, verification, and evaluation.
"""

import os
import json
import time
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

from config import (
    PretrainedModelConfig, MethodConfig, TrainConfig,
    PRETRAINED_MODELS, METHOD_CONFIGS, EXPERIMENT_CONFIGS, LoRAConfig,
)
from data import prepare_all_domains, create_dataloader
from methods import create_method, AnchorAVRDiscrete
from evaluate import evaluate_all_phases


def load_pretrained_model(model_config: PretrainedModelConfig, device: str = "cuda"):
    """Load a pretrained model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading pretrained model: {model_config.display_name}")
    print(f"  HF ID: {model_config.hf_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_id,
        dtype=torch.float32,  # Use fp32 for stability
        trust_remote_code=model_config.trust_remote_code,
    )

    model = model.to(device)
    print(f"  Model loaded. Total params: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def attach_lora(
    model,
    lora_config: LoRAConfig,
    model_config: PretrainedModelConfig,
):
    """Attach LoRA adapters to the pretrained model. Freeze all base weights."""
    # Use model-specific target modules if available
    target_modules = lora_config.target_modules or model_config.lora_target_modules

    peft_config = LoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA attached. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def train_single_phase(
    model,
    method,
    phase_key: str,
    train_dataset,
    val_datasets: dict,
    train_config: TrainConfig,
    device: str = "cuda",
):
    """Train on a single phase using the specified CL method."""
    print(f"\n{'='*60}")
    print(f"Training Phase {phase_key}")
    print(f"{'='*60}")

    method.on_phase_start(model, phase_key, val_datasets)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Only optimize trainable (LoRA) params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    n_steps = len(train_loader) * train_config.epochs_per_phase
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.warmup_steps,
        num_training_steps=n_steps,
    )

    global_step = 0
    total_loss = 0.0
    phase_start_time = time.time()

    for epoch in range(train_config.epochs_per_phase):
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            model.train()

            loss, metrics = method.compute_loss(model, batch, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            total_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Discrete mode: verify and repair periodically
            if isinstance(method, AnchorAVRDiscrete) and method.should_verify():
                method.verify_and_repair(model, device)

            if global_step % train_config.log_interval == 0:
                avg_loss = total_loss / global_step
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}/{n_steps} | Loss: {loss.item():.4f} | "
                      f"Avg: {avg_loss:.4f} | LR: {lr:.2e}")
                if metrics:
                    for k, v in metrics.items():
                        if k != "lm_loss" and v > 0:
                            print(f"    {k}: {v:.6f}")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{train_config.epochs_per_phase} | "
              f"Avg Loss: {avg_epoch_loss:.4f}")

    phase_time = time.time() - phase_start_time

    # Save anchors / compute Fisher after phase
    method.on_phase_end(model, phase_key, train_dataset)

    # Evaluate on all domains seen so far
    eval_results = evaluate_all_phases(
        model, val_datasets, method.completed_phases, device, train_config.eval_samples
    )

    eval_results["training_time"] = phase_time
    eval_results["total_steps"] = global_step
    eval_results["avg_loss"] = total_loss / max(global_step, 1)

    if isinstance(method, AnchorAVRDiscrete):
        eval_results["total_repairs"] = method.total_repairs

    return eval_results


def run_experiment(
    model_name: str,
    method_name: str,
    experiment_type: str,
    train_config: TrainConfig = None,
    lora_config: LoRAConfig = None,
    seed: int = 42,
):
    """
    Run a complete experiment: load model, attach LoRA, train sequentially.
    """
    if train_config is None:
        train_config = TrainConfig()
    if lora_config is None:
        lora_config = LoRAConfig()

    torch.manual_seed(seed)
    random.seed(seed)

    model_config = PRETRAINED_MODELS[model_name]
    method_config = METHOD_CONFIGS[method_name]
    domains = EXPERIMENT_CONFIGS[experiment_type]

    device = train_config.device

    print(f"\n{'#'*70}")
    print(f"# Experiment: {model_config.display_name} + {method_config.display_name}")
    print(f"# Domains: {experiment_type} ({len(domains)} phases)")
    print(f"{'#'*70}")

    # Load pretrained model
    model, tokenizer = load_pretrained_model(model_config, device)

    # Attach LoRA
    model = attach_lora(model, lora_config, model_config)

    # Prepare data
    phases_data, tokenizer = prepare_all_domains(
        domains, model_config.hf_id, train_config, seed=seed
    )

    # Create method
    method = create_method(method_name, method_config, train_config)

    # Validation datasets for all phases (for evaluation after each phase)
    val_datasets = {}
    for pk, pd in phases_data.items():
        val_datasets[pk] = pd["val"]

    # Train sequentially on each phase
    all_results = {
        "model": model_name,
        "method": method_name,
        "experiment": experiment_type,
        "phases": {},
    }

    for phase_key in domains.keys():
        train_dataset = phases_data[phase_key]["train"]

        phase_results = train_single_phase(
            model, method, phase_key,
            train_dataset, val_datasets,
            train_config, device,
        )

        all_results["phases"][phase_key] = phase_results

        # Print phase summary
        print(f"\n  Phase {phase_key} Summary:")
        for eval_pk, ppl in phase_results.get("perplexity", {}).items():
            print(f"    {eval_pk} perplexity: {ppl:.2f}")

    # Add storage info
    all_results["storage_kb"] = method.get_storage_kb()
    all_results["method_details"] = {
        "total_repairs": getattr(method, "total_repairs", 0),
    }

    # Save results
    os.makedirs(train_config.results_dir, exist_ok=True)
    result_file = os.path.join(
        train_config.results_dir,
        f"results_{model_name}_{method_name}_{experiment_type}.json"
    )
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {result_file}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {model_config.display_name} + {method_config.display_name}")
    print(f"{'='*60}")

    # Get final perplexity on all phases
    last_phase_key = list(domains.keys())[-1]
    final_ppls = all_results["phases"][last_phase_key].get("perplexity", {})
    for pk, ppl in final_ppls.items():
        print(f"  Phase {pk}: {ppl:.2f}")

    # Calculate forgetting factor
    forgetting_factors = {}
    for pk in domains.keys():
        after_learn_ppl = all_results["phases"][pk].get("perplexity", {}).get(pk, None)
        after_all_ppl = final_ppls.get(pk, None)
        if after_learn_ppl and after_all_ppl and after_learn_ppl > 0:
            forgetting_factors[pk] = after_all_ppl / after_learn_ppl

    if forgetting_factors:
        avg_forgetting = sum(forgetting_factors.values()) / len(forgetting_factors)
        print(f"  Avg Forgetting Factor: {avg_forgetting:.2f}x")
        for pk, ff in forgetting_factors.items():
            print(f"    Phase {pk}: {ff:.2f}x")

    print(f"  Storage: {all_results['storage_kb']:.1f} KB")

    # Clean up GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_results


# Need to import random for seed
import random
