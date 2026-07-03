"""
avr-cl: continual post-training framework.

LEARN → VERIFY → REPAIR, each phase pluggable. AVR v1 is the default
instance (SFT + PPL-ratio + snapshot-interp). v2 research (subspace
repair, DPO/GRPO, oracles, consolidators) slots in without API changes.

Quickstart:
    pip install avr-cl
    avr train configs/trace_lfm350m.yaml

Or programmatically:
    from avr.strategy import AVRStrategy
    strategy = AVRStrategy(config)
    state = strategy.run(model, tokenizer, tasks)
"""

from .framework import (
    ContinualPostTrainer, StreamState, TaskSpec, DriftInfo, VerificationResult,
    LearnStrategy, DriftDetector, RepairOperator, Oracle, Consolidator,
    NoopOracle, NoopConsolidator,
    get_lora_state, set_lora_state,
)
from .operators import SnapshotInterp, SubspaceSnapshotInterp, get_operator
from .detectors import PPLRatioDetector, get_detector
from .trainer import SFTStrategy, ReplaySFTStrategy
from .strategy import AVRStrategy
from .metrics import compute_metrics, evaluate_task_accuracy, score_answer
from .data import (load_stream, load_trace, load_mmlu_stream,
                   load_realworld_stream)

__version__ = "0.1.0"

__all__ = [
    # Core framework
    "ContinualPostTrainer", "StreamState", "TaskSpec", "DriftInfo",
    "VerificationResult",
    # Interfaces
    "LearnStrategy", "DriftDetector", "RepairOperator",
    "Oracle", "Consolidator",
    # Defaults / noops
    "NoopOracle", "NoopConsolidator",
    # Snapshot helpers
    "get_lora_state", "set_lora_state",
    # v1 implementations
    "SnapshotInterp", "PPLRatioDetector", "SFTStrategy", "ReplaySFTStrategy",
    # v2 stubs
    "SubspaceSnapshotInterp",
    # Orchestration
    "AVRStrategy",
    # Metrics + data
    "compute_metrics", "evaluate_task_accuracy", "score_answer",
    "load_stream", "load_trace", "load_mmlu_stream", "load_realworld_stream",
]
