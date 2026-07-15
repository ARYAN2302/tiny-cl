"""Model loading, LoRA setup, and chat template wrapping."""
import torch
from typing import Optional


def load_model(model_id: str, lora_rank: int = 128, lora_alpha: int = 128,
               lora_targets: list = None, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"  Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa")

    if lora_targets is None:
        lora_targets = detect_lora_targets(model)
        print(f"  Auto-detected LoRA targets: {lora_targets}", flush=True)

    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=lora_targets, bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)", flush=True)
    return model, tokenizer


def detect_lora_targets(model):
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",
                  "in_proj", "out_proj", "conv1d"]
    found = []
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c) and c not in found:
                found.append(c)
    return found if found else ["q_proj", "v_proj"]


def format_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {question}\nAssistant:"


def format_example(tokenizer, question: str, answer: str) -> str:
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        text = f"User: {question}\nAssistant: {answer}"
    return text + tokenizer.eos_token
