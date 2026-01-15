"""Abstract base class for steering vector classifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

import torch
from torch import Tensor

from steerex import ModelBackend


@dataclass
class ClassificationResult:
    """Result from a single classification."""

    score: float
    prompt: str
    layer: int
    metadata: dict = field(default_factory=dict)


class BaseClassifier(ABC):
    """
    Abstract base class for steering vector classifiers.

    All classifiers must implement:
    - name: A descriptive name for logging
    - classify(): Score a single prompt
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return classifier name for logging."""
        ...

    @abstractmethod
    def classify(
        self,
        backend: ModelBackend,
        prompt: str,
        steering_vector: Tensor,
        layer: int,
    ) -> ClassificationResult:
        """
        Classify a single prompt.

        Args:
            backend: Model backend for forward passes.
            prompt: Input prompt (already formatted with chat template).
            steering_vector: The steering vector to use.
            layer: Layer to extract activations from / apply steering at.

        Returns:
            ClassificationResult with score and metadata.
        """
        ...

    def classify_batch(
        self,
        backend: ModelBackend,
        prompts: List[str],
        steering_vector: Tensor,
        layer: int,
        batch_size: int = 8,
    ) -> List[ClassificationResult]:
        """
        Classify multiple prompts with batching for efficiency.

        Default implementation processes sequentially. Subclasses can override
        for true batched inference.

        Args:
            backend: Model backend.
            prompts: List of formatted prompts.
            steering_vector: The steering vector.
            layer: Layer for activations/steering.
            batch_size: Number of prompts per batch (for future batched impl).

        Returns:
            List of ClassificationResults.
        """
        results = []
        for prompt in prompts:
            result = self.classify(backend, prompt, steering_vector, layer)
            results.append(result)
        return results
