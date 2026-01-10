"""GPQA evaluation module."""

from src.evals.gpqa.dataset import (
    GPQASample,
    load_gpqa_dataset,
    format_gpqa_prompt,
    GPQA_SYSTEM_PROMPT,
)
from src.evals.gpqa.evaluation import GPQAEvaluator

__all__ = [
    "GPQASample",
    "load_gpqa_dataset",
    "format_gpqa_prompt",
    "GPQA_SYSTEM_PROMPT",
    "GPQAEvaluator",
]
