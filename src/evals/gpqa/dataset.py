"""GPQA dataset loading and formatting utilities."""

import random
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset


# System prompt for GPQA evaluation
GPQA_SYSTEM_PROMPT = (
    "Be concise in your reasoning. At the end of your response, return your final answer in the format "
    '"ANSWER: <answer>" where <answer> is one of A, B, C, or D.'
)


@dataclass
class GPQASample:
    """A formatted GPQA sample ready for evaluation."""

    index: int
    question: str
    choices: List[str]  # [A, B, C, D] in presentation order
    correct_answer: str  # "A", "B", "C", or "D"
    correct_text: str  # Full text of correct answer
    prompt: str  # Formatted MCQ prompt


def load_gpqa_dataset(
    subset: str = "gpqa_diamond",
    n_samples: Optional[int] = None,
    seed: int = 42,
    shuffle_choices: bool = True,
) -> List[GPQASample]:
    """
    Load GPQA dataset and format as MCQ samples.

    Args:
        subset: Dataset subset ("gpqa_diamond", "gpqa_main", "gpqa_extended").
        n_samples: Number of samples to use. None = all samples.
        seed: Random seed for reproducibility.
        shuffle_choices: Whether to shuffle choice order (recommended to avoid position bias).

    Returns:
        List of GPQASample objects ready for evaluation.
    """
    random.seed(seed)

    dataset = load_dataset("Idavidrein/gpqa", subset)["train"]

    # Sample if requested
    if n_samples is not None and n_samples < len(dataset):
        indices = random.sample(range(len(dataset)), n_samples)
        dataset = dataset.select(indices)

    samples = []
    for idx, row in enumerate(dataset):
        sample = _format_gpqa_row(row, idx, shuffle_choices, seed + idx)
        samples.append(sample)

    return samples


def _format_gpqa_row(
    row: dict,
    index: int,
    shuffle_choices: bool,
    seed: int,
) -> GPQASample:
    """Format a single GPQA row into a GPQASample."""
    random.seed(seed)

    question = row["Question"]
    correct_text = row["Correct Answer"]

    # Build choices list
    choices_with_labels = [
        (correct_text, True),
        (row["Incorrect Answer 1"], False),
        (row["Incorrect Answer 2"], False),
        (row["Incorrect Answer 3"], False),
    ]

    if shuffle_choices:
        random.shuffle(choices_with_labels)

    # Find which letter is correct
    choices = [c[0] for c in choices_with_labels]
    correct_idx = next(i for i, (_, is_correct) in enumerate(choices_with_labels) if is_correct)
    correct_answer = ["A", "B", "C", "D"][correct_idx]

    # Format the prompt
    prompt = format_gpqa_prompt(question, choices)

    return GPQASample(
        index=index,
        question=question,
        choices=choices,
        correct_answer=correct_answer,
        correct_text=correct_text,
        prompt=prompt,
    )


def format_gpqa_prompt(question: str, choices: List[str]) -> str:
    """
    Format a GPQA question as an MCQ prompt.

    Args:
        question: The question text.
        choices: List of 4 answer choices.

    Returns:
        Formatted prompt string.
    """
    labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(f"({label}) {choice}" for label, choice in zip(labels, choices))

    return f"""Answer the following multiple choice question. Think step by step before providing your final answer.

{question}

{choices_text}"""
