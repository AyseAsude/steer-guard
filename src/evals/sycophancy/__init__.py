"""Sycophancy evaluation module."""

from src.evals.sycophancy.dataset import (
    SycophancySample,
    SycophancyContrastivePair,
    load_sycophancy_dataset,
    create_sycophancy_contrastive_pairs,
    SYCOPHANCY_SYSTEM_PROMPT,
)
from src.evals.sycophancy.evaluation import LLMJudgeSycophancyEvaluator

__all__ = [
    "SycophancySample",
    "SycophancyContrastivePair",
    "load_sycophancy_dataset",
    "create_sycophancy_contrastive_pairs",
    "SYCOPHANCY_SYSTEM_PROMPT",
    "LLMJudgeSycophancyEvaluator",
]


