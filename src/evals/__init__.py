"""Evaluation pipelines for various benchmarks.

Available evaluators:
- GPQAEvaluator: 4-choice MCQ accuracy (GPQA dataset)
- MisalignmentEvaluator: Binary A/B alignment rate (discourse-grounded-misalignment-evals)
- LLMJudgeSycophancyEvaluator: Sycophancy rate using LLM-as-judge (sycophancy-eval)

Usage:
    from src.evals import GPQAEvaluator, load_gpqa_dataset

    evaluator = GPQAEvaluator(backend, tokenizer)
    samples = load_gpqa_dataset(n_samples=100)
    result = evaluator.evaluate(samples)
    print(result)
"""

from src.evals.base import (
    BaseEvaluator,
    EvalSample,
    SampleResult,
    EvaluationResult,
    print_evaluation_summary,
    compare_evaluations,
)

from src.evals.gpqa import (
    GPQAEvaluator,
    GPQASample,
    load_gpqa_dataset,
    GPQA_SYSTEM_PROMPT,
)

from src.evals.misalignment import (
    MisalignmentEvaluator,
    MisalignmentSample,
    ContrastivePair,
    load_misalignment_dataset,
    create_contrastive_pairs,
    MISALIGNMENT_SYSTEM_PROMPT,
)

from src.evals.sycophancy import (
    LLMJudgeSycophancyEvaluator,
    SycophancySample,
    SycophancyContrastivePair,
    load_sycophancy_dataset,
    create_sycophancy_contrastive_pairs,
    SYCOPHANCY_SYSTEM_PROMPT,
)

__all__ = [
    # Base
    "BaseEvaluator",
    "EvalSample",
    "SampleResult",
    "EvaluationResult",
    "print_evaluation_summary",
    "compare_evaluations",
    # GPQA
    "GPQAEvaluator",
    "GPQASample",
    "load_gpqa_dataset",
    "GPQA_SYSTEM_PROMPT",
    # Misalignment
    "MisalignmentEvaluator",
    "MisalignmentSample",
    "ContrastivePair",
    "load_misalignment_dataset",
    "create_contrastive_pairs",
    "MISALIGNMENT_SYSTEM_PROMPT",
    # Sycophancy
    "LLMJudgeSycophancyEvaluator",
    "SycophancySample",
    "SycophancyContrastivePair",
    "load_sycophancy_dataset",
    "create_sycophancy_contrastive_pairs",
    "SYCOPHANCY_SYSTEM_PROMPT",
]

