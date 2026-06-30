"""
Tiny GPT-2 model creation using HuggingFace transformers.
"""

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from config import ModelConfig


def create_model(model_config: ModelConfig) -> GPT2LMHeadModel:
    """Create a tiny GPT-2 model from a ModelConfig."""
    
    config = GPT2Config(
        vocab_size=model_config.vocab_size,
        n_positions=model_config.context_length,
        n_embd=model_config.n_embd,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_inner=model_config.n_ff,
        activation_function="gelu_new",
        resid_pdrop=model_config.dropout,
        embd_pdrop=model_config.dropout,
        attn_pdrop=model_config.dropout,
        summary_first_dropout=model_config.dropout,
        # Weight tying: share embedding and output projection
        tie_word_embeddings=True,
    )
    
    model = GPT2LMHeadModel(config)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_config.name} | Total params: {total_params:,} "
          f"({total_params/1e6:.1f}M) | Trainable: {trainable_params:,}")
    
    return model


def count_parameters(model: torch.nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_M": f"{total/1e6:.1f}M",
        "trainable_M": f"{trainable/1e6:.1f}M",
    }


def freeze_bottom_layers(model: GPT2LMHeadModel, n_layers: int) -> None:
    """Freeze the bottom N transformer layers."""
    for i in range(n_layers):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False
    
    info = count_parameters(model)
    print(f"Froze bottom {n_layers} layers. Trainable params: {info['trainable_M']} / {info['total_M']}")


def unfreeze_all(model: GPT2LMHeadModel) -> None:
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_layer_params(model: GPT2LMHeadModel, layer_idx: int) -> list:
    """Get parameter list for a specific transformer layer."""
    return list(model.transformer.h[layer_idx].parameters())
