"""Analysis utilities for sandbagging detection."""

from dataclasses import dataclass
from typing import List, Optional

from src.sandbagging.dataset import GPQASample
from src.sandbagging.evaluation import EvaluationResult, SampleResult


@dataclass
class DivergenceSample:
    """A sample where two models gave different answers."""

    sample: GPQASample
    model_a_answer: Optional[str]
    model_b_answer: Optional[str]
    model_a_correct: bool
    model_b_correct: bool
    correct_answer: str

    def __repr__(self) -> str:
        return (
            f"DivergenceSample(idx={self.sample.index}, "
            f"correct={self.correct_answer}, "
            f"model_a={self.model_a_answer}({'✓' if self.model_a_correct else '✗'}), "
            f"model_b={self.model_b_answer}({'✓' if self.model_b_correct else '✗'}))"
        )


def find_divergence_samples(
    samples: List[GPQASample],
    result_a: EvaluationResult,
    result_b: EvaluationResult,
    a_wrong_b_right: bool = True,
) -> List[DivergenceSample]:
    """
    Find samples where two models disagree.

    Args:
        samples: Original GPQA samples.
        result_a: Evaluation results from model A.
        result_b: Evaluation results from model B.
        a_wrong_b_right: If True, find samples where A is wrong and B is right.
                         If False, find all divergence samples.

    Returns:
        List of DivergenceSample objects.
    """
    # Build lookup by index
    results_a_map = {r.index: r for r in result_a.results}
    results_b_map = {r.index: r for r in result_b.results}

    divergences = []

    for sample in samples:
        res_a = results_a_map.get(sample.index)
        res_b = results_b_map.get(sample.index)

        if res_a is None or res_b is None:
            continue

        # Check for divergence
        if a_wrong_b_right:
            # Model A wrong, Model B right
            if not res_a.is_correct and res_b.is_correct:
                divergences.append(
                    DivergenceSample(
                        sample=sample,
                        model_a_answer=res_a.model_answer,
                        model_b_answer=res_b.model_answer,
                        model_a_correct=res_a.is_correct,
                        model_b_correct=res_b.is_correct,
                        correct_answer=sample.correct_answer,
                    )
                )
        else:
            # Any divergence (different answers)
            if res_a.model_answer != res_b.model_answer:
                divergences.append(
                    DivergenceSample(
                        sample=sample,
                        model_a_answer=res_a.model_answer,
                        model_b_answer=res_b.model_answer,
                        model_a_correct=res_a.is_correct,
                        model_b_correct=res_b.is_correct,
                        correct_answer=sample.correct_answer,
                    )
                )

    return divergences


def print_divergence_summary(
    divergences: List[DivergenceSample],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    show_samples: int = 5,
) -> None:
    """Print summary of divergence analysis."""
    print(f"\n{'='*60}")
    print(f"Divergence Analysis: {model_a_name} vs {model_b_name}")
    print(f"{'='*60}")
    print(f"Total divergence samples: {len(divergences)}")

    if divergences:
        a_wrong_b_right = sum(1 for d in divergences if not d.model_a_correct and d.model_b_correct)
        a_right_b_wrong = sum(1 for d in divergences if d.model_a_correct and not d.model_b_correct)
        both_wrong = sum(1 for d in divergences if not d.model_a_correct and not d.model_b_correct)

        print(f"  {model_a_name} wrong, {model_b_name} right: {a_wrong_b_right}")
        print(f"  {model_a_name} right, {model_b_name} wrong: {a_right_b_wrong}")
        print(f"  Both wrong (different answers): {both_wrong}")

        if show_samples > 0:
            print(f"\nFirst {min(show_samples, len(divergences))} divergence samples:")
            for div in divergences[:show_samples]:
                print(f"  - Index {div.sample.index}: correct={div.correct_answer}, "
                      f"{model_a_name}={div.model_a_answer}, {model_b_name}={div.model_b_answer}")
