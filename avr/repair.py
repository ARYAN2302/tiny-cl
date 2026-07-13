"""REPAIR phase: LoRA state management + weight interpolation."""
import math
import torch
import torch.nn.init as init


def get_lora_state(model):
    return {n: p.data.cpu().clone() for n, p in model.named_parameters() if "lora_" in n}

def set_lora_state(model, state, device="cuda"):
    for n, p in model.named_parameters():
        if "lora_" in n and n in state:
            p.data.copy_(state[n].to(device).to(p.data.dtype))

def reset_lora(model):
    for n, p in model.named_parameters():
        if "lora_A" in n:
            init.kaiming_uniform_(p.data, a=math.sqrt(5))
        elif "lora_B" in n:
            p.data.zero_()

def repair(model, snapshot, alpha=0.1, device="cuda"):
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name and name in snapshot:
            p.data.copy_((1.0 - alpha) * p.data + alpha * snapshot[name].to(device))
            n += 1
    return n
