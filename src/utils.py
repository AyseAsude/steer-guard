"""Utility functions for steering vector experiments."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from torch import Tensor

from steering_vectors.core import TrainingDatapoint


def load_misalignment_dataset(
    split: str | None = None,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Load the misalignment evaluation dataset.

    Dataset: geodesic-research/discourse-grounded-misalignment-evals

    Contains 4,174 questions across 2 splits:
    - article_questions (2.67k rows)
    - textbook_questions (1.5k rows)

    Each row has: passage, question, choices (2 options),
    misaligned_choice (0 or 1), topics

    Args:
        split: Optional split name ("article_questions" or "textbook_questions").
               If None, loads all data.
        cache_dir: Optional directory to cache the dataset.

    Returns:
        HuggingFace Dataset object.
    """
    dataset_name = "geodesic-research/discourse-grounded-misalignment-evals"

    if split:
        return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


    article = load_dataset(dataset_name, split="article_questions", cache_dir=cache_dir)
    textbook = load_dataset(dataset_name, split="textbook_questions", cache_dir=cache_dir)
    return concatenate_datasets([article, textbook])


def load_alpaca_dataset(
    n_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> Dataset:
    """Load the Alpaca dataset for control/benign prompts.

    Dataset: tatsu-lab/alpaca

    Args:
        n_samples: Optional number of samples to randomly select.
        seed: Random seed for sampling.
        cache_dir: Optional directory to cache the dataset.

    Returns:
        HuggingFace Dataset object.
    """
    dataset = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)

    if n_samples is not None and n_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(n_samples))

    return dataset


def create_contrastive_datapoints(
    dataset: Dataset,
    prompt_template: str = "{passage}\n\nQuestion: {question}\n\nAnswer:",
    promote_misaligned: bool = True,
) -> list[TrainingDatapoint]:
    """Convert misalignment dataset rows to TrainingDatapoint pairs.

    Creates contrastive pairs where one completion is promoted and the other
    is suppressed.

    Args:
        dataset: Misalignment dataset with passage, question, choices,
                 misaligned_choice fields.
        prompt_template: Format string with {passage}, {question} placeholders.
        promote_misaligned: If True, promote misaligned choice and suppress aligned.
                           If False, promote aligned and suppress misaligned.

    Returns:
        List of TrainingDatapoint objects for steering vector training.
    """
    datapoints = []

    for row in dataset:
        prompt = prompt_template.format(
            passage=row["passage"],
            question=row["question"],
        )

        misaligned_idx = row["misaligned_choice"]
        aligned_idx = 1 - misaligned_idx

        misaligned_completion = row["choices"][misaligned_idx]
        aligned_completion = row["choices"][aligned_idx]

        if promote_misaligned:
            dst_completions = [misaligned_completion]
            src_completions = [aligned_completion]
        else:
            dst_completions = [aligned_completion]
            src_completions = [misaligned_completion]

        datapoints.append(TrainingDatapoint(
            prompt=prompt,
            dst_completions=dst_completions,
            src_completions=src_completions,
        ))

    return datapoints


def compute_kl_divergence(
    logits_p: Tensor,
    logits_q: Tensor,
    dim: int = -1,
) -> Tensor:
    """Compute KL divergence between two distributions.

    KL(P || Q) = sum(P * log(P / Q))

    Args:
        logits_p: Logits for reference distribution P.
        logits_q: Logits for comparison distribution Q.
        dim: Dimension to compute softmax over.

    Returns:
        KL divergence tensor. Same shape as input minus the dim dimension.
    """
    log_p = torch.log_softmax(logits_p, dim=dim)
    log_q = torch.log_softmax(logits_q, dim=dim)
    p = torch.softmax(logits_p, dim=dim)

    kl = (p * (log_p - log_q)).sum(dim=dim)
    return kl


def compute_batch_kl_divergence(
    logits_p: Tensor,
    logits_q: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """Compute KL divergence for a batch, with optional reduction.

    Args:
        logits_p: Reference logits of shape (batch, seq_len, vocab_size).
        logits_q: Comparison logits of shape (batch, seq_len, vocab_size).
        reduction: "mean", "sum", or "none".

    Returns:
        Reduced KL divergence or per-token KL if reduction="none".
    """
    kl = compute_kl_divergence(logits_p, logits_q, dim=-1)

    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    return kl


def save_vector(
    vector: Tensor,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a steering vector with metadata.

    Saves two files:
    - {path}.pt: The vector tensor
    - {path}.json: Metadata (training config, timestamp, etc.)

    Args:
        vector: Steering vector tensor.
        path: Base path (without extension).
        metadata: Optional metadata dictionary.

    Returns:
        Path to the saved vector file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vector_path = path.with_suffix(".pt")
    meta_path = path.with_suffix(".json")

    torch.save(vector, vector_path)

    meta = metadata or {}
    meta["saved_at"] = datetime.now().isoformat()
    meta["shape"] = list(vector.shape)
    meta["dtype"] = str(vector.dtype)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return vector_path


def load_vector(path: str | Path) -> tuple[Tensor, dict[str, Any]]:
    """Load a steering vector and its metadata.

    Args:
        path: Path to the vector file (.pt) or base path.

    Returns:
        Tuple of (vector tensor, metadata dict).
    """
    path = Path(path)

    if path.suffix != ".pt":
        path = path.with_suffix(".pt")

    vector = torch.load(path, weights_only=True)

    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return vector, metadata


def format_alpaca_prompt(row: dict, template: str | None = None) -> str:
    """Format an Alpaca dataset row into a prompt.

    Args:
        row: Alpaca dataset row with 'instruction', 'input', 'output' fields.
        template: Optional custom template. Uses default if None.

    Returns:
        Formatted prompt string.
    """
    if template:
        return template.format(**row)

    if row.get("input"):
        return f"Instruction: {row['instruction']}\n\nInput: {row['input']}\n\nResponse:"
    return f"Instruction: {row['instruction']}\n\nResponse:"
