"""Sandbagging detection experiments module."""

from src.sandbagging.dataset import (
    load_gpqa_dataset,
    format_gpqa_prompt,
    GPQASample,
    GPQA_SYSTEM_PROMPT,
)
from src.sandbagging.evaluation import (
    evaluate_accuracy,
    evaluate_with_steering,
    EvaluationResult,
)
from src.sandbagging.analysis import (
    find_divergence_samples,
    DivergenceSample,
)

__all__ = [
    # Dataset
    "load_gpqa_dataset",
    "format_gpqa_prompt",
    "GPQASample",
    "GPQA_SYSTEM_PROMPT",
    # Evaluation
    "evaluate_accuracy",
    "evaluate_with_steering",
    "EvaluationResult",
    # Analysis
    "find_divergence_samples",
    "DivergenceSample",
]
