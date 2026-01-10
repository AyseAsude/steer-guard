"""GPQA evaluation using the template pattern."""

import re
from typing import Optional

from src.evals.base import BaseEvaluator
from src.evals.gpqa.dataset import GPQASample, GPQA_SYSTEM_PROMPT
from src.utils import apply_chat_template


class GPQAEvaluator(BaseEvaluator):
    """
    Evaluator for GPQA dataset (4-choice MCQ).

    Measures accuracy: percentage of correctly answered questions.
    """

    def __init__(
        self,
        backend,
        tokenizer,
        system_prompt: Optional[str] = GPQA_SYSTEM_PROMPT,
    ):
        """
        Initialize GPQA evaluator.

        Args:
            backend: HuggingFaceBackend instance.
            tokenizer: Tokenizer for chat template.
            system_prompt: System prompt (defaults to GPQA_SYSTEM_PROMPT).
        """
        super().__init__(backend, tokenizer, system_prompt)

    def get_metric_name(self) -> str:
        return "accuracy"

    def get_expected_answer(self, sample: GPQASample) -> str:
        return sample.correct_answer

    def format_prompt(self, sample: GPQASample) -> str:
        """Apply chat template to the sample's prompt."""
        return apply_chat_template(
            self.tokenizer,
            sample.prompt,
            system_prompt=self.system_prompt,
        )

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer letter from model response.

        Handles various response formats (in order of priority):
        1. "ANSWER: A" format (preferred, from system prompt instruction)
        2. "The answer is A" or similar patterns

        Args:
            response: Raw model response.

        Returns:
            "A", "B", "C", "D", or None if cannot extract.
        """
        response_upper = response.strip().upper()

        # Priority 1: Look for "ANSWER: X" format (from system prompt instruction)
        # Search from the end of response since that's where we asked for it
        answer_pattern = r"ANSWER:\s*\(?([ABCD])\)?"
        matches = list(re.finditer(answer_pattern, response_upper))
        if matches:
            # Take the last match (final answer)
            return matches[-1].group(1)

        # Priority 2: Check for patterns like "THE ANSWER IS X"
        for pattern in ["THE ANSWER IS ", "ANSWER IS ", "FINAL ANSWER: ", "FINAL ANSWER IS "]:
            if pattern in response_upper:
                after = response_upper.split(pattern)[-1].strip()
                if after and after[0] in "ABCD":
                    return after[0]

        return None

    def check_correct(self, sample: GPQASample, answer: Optional[str]) -> bool:
        """Check if model's answer matches the correct answer."""
        if answer is None:
            return False
        return answer == sample.correct_answer
