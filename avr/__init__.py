"""
avr-cl: continual post-training framework.

LEARN → VERIFY → REPAIR. Detects when fine-tuning broke old capabilities
and repairs them in weight space. No replay, no gradients at repair time.

Quickstart:
    import avr
    result = avr.run(
        model="Qwen/Qwen3-1.7B",
        tasks=[
            ("gsm8k", train_pairs, eval_pairs),
            ("math", train_pairs, eval_pairs),
        ],
        lora_rank=128,
    )
    print(f"BWT: {result['bwt']:+.3f}  Repairs: {result['repairs']}")
"""
from .run import run, load_model, evaluate, compute_metrics
from .run import generate_batch, default_scorer, normalize_answer

__version__ = "0.1.0"

__all__ = [
    "run",          # Main entry: avr.run(model, tasks, ...)
    "load_model",   # Load any HF model + LoRA
    "evaluate",     # Evaluate on a task
    "compute_metrics",  # BWT, FF, ACC from R-matrix
    "generate_batch",   # Batched generation
    "default_scorer",   # Default answer scorer
    "normalize_answer", # Text normalization
]
