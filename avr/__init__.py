"""
avr-cl: continual post-training with drift detection + repair.

LEARN → VERIFY → REPAIR. Each phase is a separate module you can swap:
  - avr.learn: train_sft, consolidate (LEARN)
  - avr.verify: compute_ppl, eval_ppls, check_drift (VERIFY)
  - avr.repair: get_lora_state, set_lora_state, reset_lora, repair (REPAIR)
  - avr.eval: evaluate, generate_batch, default_scorer (evaluation)
  - avr.model: load_model, format_prompt, format_example (model handling)

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
from .run import run, compute_metrics
from .model import load_model, detect_lora_targets, format_prompt, format_example
from .learn import train_sft, consolidate
from .verify import compute_ppl, eval_ppls, check_drift
from .repair import get_lora_state, set_lora_state, reset_lora, repair
from .eval import evaluate, generate_batch, default_scorer, normalize_answer

__version__ = "0.1.0"

__all__ = [
    "run",
    "compute_metrics",
    "load_model", "detect_lora_targets", "format_prompt", "format_example",
    "train_sft", "consolidate",
    "compute_ppl", "eval_ppls", "check_drift",
    "get_lora_state", "set_lora_state", "reset_lora", "repair",
    "evaluate", "generate_batch", "default_scorer", "normalize_answer",
]
