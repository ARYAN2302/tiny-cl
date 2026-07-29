"""T4-friendly model loading and generation."""
from __future__ import annotations

import torch


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load_learner(model_id: str = DEFAULT_MODEL, lora_rank: int = 16):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        device_map="auto",
        dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.config.use_cache = False
    return model, tokenizer


def prompt_text(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except (AttributeError, ValueError):
        return f"User: {user_prompt}\nAssistant:"


@torch.inference_mode()
def answer(model, tokenizer, user_prompt: str, max_new_tokens: int = 256) -> str:
    device = next(model.parameters()).device
    text = prompt_text(tokenizer, user_prompt)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    was_training = model.training
    model.eval()
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    if was_training:
        model.train()
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
