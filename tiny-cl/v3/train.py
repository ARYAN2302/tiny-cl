"""
V3 Training Loop: The Living Model on LFM2.5-350M.
Fast-slow LoRA, AVR loop, and consolidation.
"""

import os
import json
import time
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from config import (
    ModelConfig, LivingModelConfig, FastSlowLoRAConfig, TrainConfig,
    NaiveLoRAConfig, PRETRAINED_MODELS, EXPERIMENT_CONFIGS,
    CONV_LAYER_IDS, ATTN_LAYER_IDS,
)
from data import prepare_all_domains, create_dataloader
from methods import LivingModelMethod, NaiveMethod
from evaluate import evaluate_all_phases, compute_forgetting_factors


def load_lfm2_5(model_config: ModelConfig, device: str = "cuda"):
    print(f"Loading {model_config.display_name} from {model_config.hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded. Total params: {n_params:,}")
    return model, tokenizer


def apply_fast_slow_lora(model, lora_config: FastSlowLoRAConfig, model_config: ModelConfig):
    conv_targets = []
    for idx in CONV_LAYER_IDS:
        for proj in lora_config.fast_target_modules:
            conv_targets.append(f"layers.{idx}.conv.{proj}")

    attn_targets = []
    for idx in ATTN_LAYER_IDS:
        for proj in lora_config.slow_target_modules:
            attn_targets.append(f"layers.{idx}.self_attn.{proj}")

    all_targets = conv_targets + attn_targets

    peft_config = LoraConfig(
        r=lora_config.fast_rank, lora_alpha=lora_config.fast_alpha,
        lora_dropout=lora_config.fast_dropout, target_modules=all_targets,
        bias=lora_config.bias, task_type=lora_config.task_type,
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Fast-Slow LoRA attached. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    conv_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "conv." in n)
    attn_lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "self_attn." in n)
    print(f"    Conv LoRA (fast): {conv_lora:,} params")
    print(f"    Attn LoRA (slow): {attn_lora:,} params")
    return model


def apply_naive_lora(model, lora_config: NaiveLoRAConfig, model_config: ModelConfig):
    peft_config = LoraConfig(
        r=lora_config.rank, lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout, target_modules=lora_config.target_modules,
        bias=lora_config.bias, task_type=lora_config.task_type,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Naive LoRA attached. Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def run_living_model(
    model_name: str = "lfm2.5-350M",
    experiment: str = "hero",
    train_config: TrainConfig = None,
    lora_config: FastSlowLoRAConfig = None,
    living_config: LivingModelConfig = None,
    seed: int = 42,
):
    if train_config is None: train_config = TrainConfig()
    if lora_config is None: lora_config = FastSlowLoRAConfig()
    if living_config is None: living_config = LivingModelConfig()

    torch.manual_seed(seed)
    random.seed(seed)

    model_config = PRETRAINED_MODELS[model_name]
    domains = EXPERIMENT_CONFIGS[experiment]
    device = train_config.device

    print(f"\n{'#'*70}")
    print(f"# THE LIVING MODEL: {model_config.display_name}")
    print(f"# Domains: {experiment} ({len(domains)} phases)")
    print(f"# Architecture: 10 conv (fast) + 6 attn (slow)")
    print(f"{'#'*70}")

    model, tokenizer = load_lfm2_5(model_config, device)
    model = apply_fast_slow_lora(model, lora_config, model_config)
    phases_data, tokenizer = prepare_all_domains(domains, model_config.hf_id, train_config, seed=seed)

    method = LivingModelMethod(living_config, lora_config, train_config)
    val_datasets = {pk: pd["val"] for pk, pd in phases_data.items()}

    all_results = {"model": model_name, "method": "living_model", "experiment": experiment, "phases": {}}

    for phase_key in domains.keys():
        domain_config = domains[phase_key]
        train_dataset = phases_data[phase_key]["train"]

        print(f"\n{'='*60}")
        print(f"Phase {phase_key}: {domain_config.display_name} (salience={domain_config.salience})")
        print(f"{'='*60}")

        # ABSORB
        method.on_phase_start(model, phase_key, domain_salience=domain_config.salience)

        train_loader = create_dataloader(train_dataset, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lora_config.fast_lr, weight_decay=train_config.weight_decay)
        n_steps = len(train_loader) * train_config.epochs_per_phase
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config.warmup_steps, num_training_steps=n_steps)

        global_step = 0
        total_loss = 0.0
        phase_start_time = time.time()

        for epoch in range(train_config.epochs_per_phase):
            for batch in train_loader:
                model.train()
                loss, metrics = method.compute_loss(model, batch, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                global_step += 1

                if method.phase_controller.should_verify(global_step):
                    method.verify(model, device)

                if global_step % train_config.log_interval == 0:
                    print(f"  Step {global_step}/{n_steps} | Loss: {loss.item():.4f}")

        method.on_phase_end(model, phase_key, train_dataset)

        # Eval after absorb
        after_absorb_eval = evaluate_all_phases(model, val_datasets, method.completed_phases, device, train_config.eval_samples)
        print(f"\n  After ABSORB:")
        for pk, ppl in after_absorb_eval.get("perplexity", {}).items():
            print(f"    {pk}: {ppl:.2f}")

        # VERIFY -> REPAIR -> CONSOLIDATE
        vrc_results = method.run_verify_repair_consolidate(model, phase_key, train_dataset, device)

        # Eval after consolidation
        after_consolid_eval = evaluate_all_phases(model, val_datasets, method.completed_phases, device, train_config.eval_samples)
        print(f"\n  After CONSOLIDATE:")
        for pk, ppl in after_consolid_eval.get("perplexity", {}).items():
            print(f"    {pk}: {ppl:.2f}")

        phase_time = time.time() - phase_start_time
        after_consolid_eval["training_time"] = phase_time
        after_consolid_eval["after_absorb_ppl"] = after_absorb_eval.get("perplexity", {})
        after_consolid_eval["vrc_results"] = {k: str(v) for k, v in vrc_results.items()}
        after_consolid_eval["total_repairs"] = method.total_repairs
        after_consolid_eval["total_consolidations"] = method.total_consolidations
        after_consolid_eval["health_scores"] = dict(method.anchor_store.health)
        after_consolid_eval["avg_loss"] = total_loss / max(global_step, 1)
        all_results["phases"][phase_key] = after_consolid_eval

    all_results["storage_kb"] = method.get_storage_kb()
    final_ppls = all_results["phases"][list(domains.keys())[-1]].get("perplexity", {})
    forgetting = compute_forgetting_factors(all_results, list(domains.keys()))

    print(f"\n{'='*60}")
    print(f"LIVING MODEL RESULTS: {model_config.display_name}")
    print(f"{'='*60}")
    for pk, ppl in final_ppls.items():
        ff = forgetting.get(pk, float("inf"))
        health = method.anchor_store.health.get(pk, 0.0)
        print(f"  Phase {pk}: PPL={ppl:.2f}, Forgetting={ff:.2f}x, Health={health:.3f}")

    avg_ff = sum(forgetting.values()) / len(forgetting) if forgetting else float("inf")
    print(f"  Avg Forgetting: {avg_ff:.2f}x")
    print(f"  Total Repairs: {method.total_repairs}")
    print(f"  Total Consolidations: {method.total_consolidations}")
    print(f"  Storage: {all_results['storage_kb']:.1f} KB")

    os.makedirs(train_config.results_dir, exist_ok=True)
    result_file = os.path.join(train_config.results_dir, f"living_model_{model_name}_{experiment}.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {result_file}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_results


def run_naive_baseline(
    model_name: str = "lfm2.5-350M",
    experiment: str = "hero",
    train_config: TrainConfig = None,
    lora_config: NaiveLoRAConfig = None,
    seed: int = 42,
):
    if train_config is None: train_config = TrainConfig()
    if lora_config is None: lora_config = NaiveLoRAConfig()

    torch.manual_seed(seed)
    random.seed(seed)

    model_config = PRETRAINED_MODELS[model_name]
    domains = EXPERIMENT_CONFIGS[experiment]
    device = train_config.device

    print(f"\n{'#'*70}")
    print(f"# NAIVE BASELINE: {model_config.display_name}")
    print(f"{'#'*70}")

    model, tokenizer = load_lfm2_5(model_config, device)
    model = apply_naive_lora(model, lora_config, model_config)
    phases_data, tokenizer = prepare_all_domains(domains, model_config.hf_id, train_config, seed=seed)

    method = NaiveMethod(train_config)
    val_datasets = {pk: pd["val"] for pk, pd in phases_data.items()}
    all_results = {"model": model_name, "method": "naive", "experiment": experiment, "phases": {}}

    for phase_key in domains.keys():
        train_dataset = phases_data[phase_key]["train"]
        method.on_phase_start(model, phase_key)

        train_loader = create_dataloader(train_dataset, batch_size=train_config.batch_size, shuffle=True, drop_last=True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
        n_steps = len(train_loader) * train_config.epochs_per_phase
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=train_config.warmup_steps, num_training_steps=n_steps)

        global_step = 0
        total_loss = 0.0
        for epoch in range(train_config.epochs_per_phase):
            for batch in train_loader:
                model.train()
                loss, metrics = method.compute_loss(model, batch, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, train_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                global_step += 1
                if global_step % train_config.log_interval == 0:
                    print(f"  Step {global_step}/{n_steps} | Loss: {loss.item():.4f}")

        method.on_phase_end(model, phase_key, train_dataset)
        eval_results = evaluate_all_phases(model, val_datasets, method.completed_phases, device, train_config.eval_samples)
        eval_results["avg_loss"] = total_loss / max(global_step, 1)
        all_results["phases"][phase_key] = eval_results

        print(f"  Phase {phase_key}:")
        for pk, ppl in eval_results.get("perplexity", {}).items():
            print(f"    {pk}: {ppl:.2f}")

    forgetting = compute_forgetting_factors(all_results, list(domains.keys()))
    final_ppls = all_results["phases"][list(domains.keys())[-1]].get("perplexity", {})

    print(f"\n{'='*60}")
    print(f"NAIVE BASELINE RESULTS")
    print(f"{'='*60}")
    for pk, ppl in final_ppls.items():
        ff = forgetting.get(pk, float("inf"))
        print(f"  Phase {pk}: PPL={ppl:.2f}, Forgetting={ff:.2f}x")

    os.makedirs(train_config.results_dir, exist_ok=True)
    result_file = os.path.join(train_config.results_dir, f"naive_{model_name}_{experiment}.json")
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_results
