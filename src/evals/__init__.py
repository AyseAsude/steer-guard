"""Evaluation pipelines for various benchmarks.

Available evaluators:
- GPQAEvaluator: 4-choice MCQ accuracy (GPQA dataset)
- MisalignmentEvaluator: Binary A/B alignment rate (discourse-grounded-misalignment-evals)
- SycophancyEvaluator: Non-sycophancy rate (sycophancy-eval)

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
    load_misalignment_dataset,
    MISALIGNMENT_SYSTEM_PROMPT,
)

from src.evals.sycophancy import (
    SycophancyEvaluator,
    SycophancySample,
    load_sycophancy_dataset,
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
    "load_misalignment_dataset",
    "MISALIGNMENT_SYSTEM_PROMPT",
    # Sycophancy
    "SycophancyEvaluator",
    "SycophancySample",
    "load_sycophancy_dataset",
    "SYCOPHANCY_SYSTEM_PROMPT",
]
