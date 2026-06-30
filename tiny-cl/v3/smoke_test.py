"""
V3 Smoke Test: Verify LFM2.5-350M + fast-slow LoRA works.
"""
import torch
from config import LFM2_5_CONFIG, FastSlowLoRAConfig, CONV_LAYER_IDS, ATTN_LAYER_IDS
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def smoke_test():
    print("=" * 60)
    print("V3 SMOKE TEST: LFM2.5-350M + Fast-Slow LoRA")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load model
    print("\n1. Loading LFM2.5-350M...")
    tokenizer = AutoTokenizer.from_pretrained(LFM2_5_CONFIG.hf_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LFM2_5_CONFIG.hf_id, torch_dtype=torch.float32, trust_remote_code=True,
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Loaded. Params: {n_params:,}")

    # 2. Inspect layers
    print("\n2. Inspecting layers...")
    for idx in range(16):
        layer = model.model.layers[idx]
        has_conv = hasattr(layer, 'conv')
        has_attn = hasattr(layer, 'self_attn')
        layer_type = "CONV" if has_conv and not has_attn else "ATTN" if has_attn else "UNKNOWN"
        print(f"   Layer {idx}: {layer_type}")

    # 3. Apply fast-slow LoRA
    print("\n3. Applying Fast-Slow LoRA...")
    lora_config = FastSlowLoRAConfig()
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
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 4. Forward pass
    print("\n4. Testing forward pass...")
    input_ids = tokenizer("Hello, this is a smoke test", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True)
    print(f"   Output shape: {outputs.logits.shape}")
    print(f"   Hidden states: {len(outputs.hidden_states)} layers")

    # 5. Attn hidden states
    print("\n5. Attn layer hidden states...")
    for idx in ATTN_LAYER_IDS:
        hs = outputs.hidden_states[idx + 1]
        print(f"   Layer {idx} (attn): shape={hs.shape}, mean={hs.float().mean():.4f}")

    # 6. Freeze/unfreeze test
    print("\n6. Testing freeze/unfreeze...")
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "self_attn." in name:
            param.requires_grad = False
            frozen += 1
    active = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   After freezing attn: {active} trainable, {frozen} frozen")

    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    print(f"   Conv-only training step: loss={loss.item():.4f}")

    for name, param in model.named_parameters():
        if "lora_" in name and "self_attn." in name:
            param.requires_grad = True
    active = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   After unfreezing attn: {active} trainable")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    smoke_test()
