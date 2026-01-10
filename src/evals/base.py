"""Base evaluator with template pattern for all evaluation benchmarks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any

from tqdm import tqdm

from src.utils import apply_chat_template


@dataclass
class EvalSample:
    """Base sample that all datasets extend."""

    index: int
    prompt: str  # Formatted prompt ready for chat template


@dataclass
class SampleResult:
    """Result for a single evaluated sample."""

    index: int
    expected: str  # Expected/correct answer
    actual: Optional[str]  # Model's parsed answer (None if invalid)
    is_correct: bool
    raw_response: str


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""

    metric_name: str  # e.g., "accuracy", "alignment_rate", "non_sycophancy_rate"
    metric_value: float  # Primary metric (0.0 - 1.0)
    correct_count: int
    incorrect_count: int
    invalid_count: int
    total_samples: int
    results: List[SampleResult] = field(default_factory=list)

    @property
    def invalid_rate(self) -> float:
        """Percentage of samples with unparseable responses."""
        return self.invalid_count / self.total_samples if self.total_samples > 0 else 0.0

    def get_invalid_samples(self) -> List[SampleResult]:
        """Get all samples where answer extraction failed."""
        return [r for r in self.results if r.actual is None]

    def __repr__(self) -> str:
        invalid_str = f", invalid={self.invalid_count}" if self.invalid_count > 0 else ""
        return (
            f"EvaluationResult({self.metric_name}={self.metric_value:.2%}, "
            f"correct={self.correct_count}, incorrect={self.incorrect_count}"
            f"{invalid_str}, total={self.total_samples})"
        )


class BaseEvaluator(ABC):
    """
    Template pattern base class for evaluation.

    Subclasses implement the hooks:
    - format_prompt(): Convert sample to prompt string
    - extract_answer(): Parse model response
    - check_correct(): Determine if answer is correct

    The base class handles:
    - Batched generation for GPU efficiency
    - Steering vector application
    - Result aggregation
    - Progress tracking
    """

    def __init__(
        self,
        backend,
        tokenizer,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            backend: HuggingFaceBackend instance.
            tokenizer: Tokenizer for chat template.
            system_prompt: Optional system prompt for all evaluations.
        """
        self.backend = backend
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt

    # === Template methods (don't override) ===

    def evaluate(
        self,
        samples: List[Any],
        batch_size: int = 8,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate model on samples with batched generation.

        Args:
            samples: List of dataset-specific sample objects.
            batch_size: Number of samples per batch for GPU parallelism.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling (False = greedy).
            verbose: Show progress bar.

        Returns:
            EvaluationResult with metrics and per-sample results.
        """
        results = []
        correct_count = 0
        incorrect_count = 0
        invalid_count = 0

        num_batches = (len(samples) + batch_size - 1) // batch_size
        iterator = range(0, len(samples), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc=f"Evaluating ({self.get_metric_name()})", total=num_batches)

        for batch_start in iterator:
            batch = samples[batch_start : batch_start + batch_size]

            # Format prompts
            prompts = [self.format_prompt(s) for s in batch]

            # Batch generation
            responses = self.backend.generate_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

            # Process results
            for sample, response in zip(batch, responses):
                parsed = self.extract_answer(response)
                is_correct = self.check_correct(sample, parsed)

                if parsed is None:
                    invalid_count += 1
                elif is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

                results.append(
                    SampleResult(
                        index=sample.index,
                        expected=self.get_expected_answer(sample),
                        actual=parsed,
                        is_correct=is_correct,
                        raw_response=response,
                    )
                )

        return self._aggregate(results, correct_count, incorrect_count, invalid_count)

    def evaluate_with_steering(
        self,
        samples: List[Any],
        steering_mode,
        layers: int | List[int],
        strength: float = 1.0,
        batch_size: int = 8,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate model with steering vector applied.

        Args:
            samples: List of dataset-specific sample objects.
            steering_mode: Steering mode (e.g., VectorSteering).
            layers: Layer(s) to apply steering at.
            strength: Steering strength multiplier.
            batch_size: Number of samples per batch.
            max_new_tokens: Maximum tokens to generate.
            do_sample: Whether to use sampling.
            verbose: Show progress bar.

        Returns:
            EvaluationResult with metrics and per-sample results.
        """
        results = []
        correct_count = 0
        incorrect_count = 0
        invalid_count = 0

        num_batches = (len(samples) + batch_size - 1) // batch_size
        iterator = range(0, len(samples), batch_size)
        if verbose:
            iterator = tqdm(
                iterator,
                desc=f"Evaluating steered ({self.get_metric_name()})",
                total=num_batches,
            )

        for batch_start in iterator:
            batch = samples[batch_start : batch_start + batch_size]

            # Format prompts
            prompts = [self.format_prompt(s) for s in batch]

            # Batch generation with steering
            responses = self.backend.generate_with_steering_batch(
                prompts,
                steering_mode=steering_mode,
                layers=layers,
                strength=strength,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

            # Process results
            for sample, response in zip(batch, responses):
                parsed = self.extract_answer(response)
                is_correct = self.check_correct(sample, parsed)

                if parsed is None:
                    invalid_count += 1
                elif is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

                results.append(
                    SampleResult(
                        index=sample.index,
                        expected=self.get_expected_answer(sample),
                        actual=parsed,
                        is_correct=is_correct,
                        raw_response=response,
                    )
                )

        return self._aggregate(results, correct_count, incorrect_count, invalid_count)

    def _aggregate(
        self,
        results: List[SampleResult],
        correct_count: int,
        incorrect_count: int,
        invalid_count: int,
    ) -> EvaluationResult:
        """Aggregate results into EvaluationResult."""
        total = len(results)
        valid = correct_count + incorrect_count
        metric_value = correct_count / valid if valid > 0 else 0.0

        return EvaluationResult(
            metric_name=self.get_metric_name(),
            metric_value=metric_value,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            invalid_count=invalid_count,
            total_samples=total,
            results=results,
        )

    # === Hooks (must override) ===

    @abstractmethod
    def format_prompt(self, sample: Any) -> str:
        """
        Format a sample into a prompt string with chat template applied.

        Args:
            sample: Dataset-specific sample object.

        Returns:
            Formatted prompt string ready for generation.
        """
        pass

    @abstractmethod
    def extract_answer(self, response: str) -> Optional[str]:
        """
        Parse model response into an answer.

        Args:
            response: Raw model response string.

        Returns:
            Parsed answer (e.g., "A", "B") or None if unparseable.
        """
        pass

    @abstractmethod
    def check_correct(self, sample: Any, answer: Optional[str]) -> bool:
        """
        Determine if the parsed answer is correct for this sample.

        Args:
            sample: Dataset-specific sample object.
            answer: Parsed answer from extract_answer().

        Returns:
            True if answer is correct/desired behavior.
        """
        pass

    # === Optional hooks (override if needed) ===

    def get_metric_name(self) -> str:
        """Return the name of the primary metric."""
        return "accuracy"

    def get_expected_answer(self, sample: Any) -> str:
        """Return the expected/correct answer for logging."""
        return "unknown"


def print_evaluation_summary(
    result: EvaluationResult,
    label: str = "Model",
    show_invalid_responses: bool = False,
    max_invalid_to_show: int = 5,
) -> None:
    """
    Pretty print evaluation summary.

    Args:
        result: Evaluation result to display.
        label: Label for the evaluation.
        show_invalid_responses: If True, print the raw responses that couldn't be parsed.
        max_invalid_to_show: Maximum number of invalid responses to display.
    """
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"{result.metric_name}: {result.metric_value:.1%}")
    print(f"  Correct:   {result.correct_count}")
    print(f"  Incorrect: {result.incorrect_count}")

    if result.invalid_count > 0:
        print(f"  Invalid:   {result.invalid_count} ({result.invalid_rate:.1%} of total)")
        if result.invalid_rate > 0.1:
            print(f"  WARNING: High invalid rate! Check answer extraction or model output format.")

        if show_invalid_responses:
            invalid_samples = result.get_invalid_samples()
            n_show = min(len(invalid_samples), max_invalid_to_show)
            print(f"\n  First {n_show} invalid responses:")
            for sample in invalid_samples[:n_show]:
                response_preview = sample.raw_response[:100].replace('\n', ' ')
                print(f"    [{sample.index}] \"{response_preview}...\"")
    else:
        print(f"  Invalid:   0")


def compare_evaluations(
    baseline: EvaluationResult,
    steered: EvaluationResult,
    baseline_label: str = "Baseline",
    steered_label: str = "Steered",
) -> None:
    """Compare two evaluation results side by side."""
    delta = steered.metric_value - baseline.metric_value
    metric = baseline.metric_name
    print(
        f"\n{metric}: {baseline_label}: {baseline.metric_value:.1%} | "
        f"{steered_label}: {steered.metric_value:.1%} | Delta: {delta:+.1%}"
    )
