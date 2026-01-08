"""Dot-product based steering vector classifier."""

from typing import List, Literal

import torch
from torch import Tensor

from steering_vectors import ModelBackend

from src.classifiers.base import BaseClassifier, ClassificationResult


class DotProductClassifier(BaseClassifier):
    """
    Classifier using dot product between steering vector and activations.

    Algorithm:
    1. Feed prompt through model
    2. Capture activations at layer L for all tokens
    3. Compute dot product between steering vector and each token's activation
    4. Aggregate dot products (mean/max/last) as classification score

    Higher score -> model's activations align more with steering direction.
    """

    def __init__(
        self,
        aggregation: Literal["mean", "max", "last"] = "mean",
        normalize_vector: bool = True,
    ):
        """
        Initialize DotProductClassifier.

        Args:
            aggregation: How to aggregate per-token dot products.
                - "mean": Average across all tokens
                - "max": Maximum dot product
                - "last": Use only the last token
            normalize_vector: If True, normalize steering vector before dot product.
        """
        self._aggregation = aggregation
        self._normalize_vector = normalize_vector

    @property
    def name(self) -> str:
        return f"DotProduct({self._aggregation})"

    def classify(
        self,
        backend: ModelBackend,
        prompt: str,
        steering_vector: Tensor,
        layer: int,
    ) -> ClassificationResult:
        """
        Compute dot-product classification score.

        Args:
            backend: Model backend.
            prompt: Formatted prompt string.
            steering_vector: Steering vector (hidden_dim,).
            layer: Layer to extract activations from.

        Returns:
            ClassificationResult with aggregated dot product score.
        """
        # Prepare vector on correct device/dtype
        vec = steering_vector.to(backend.get_device(), backend.get_dtype())
        if self._normalize_vector:
            vec = vec / vec.norm()

        # Storage for captured activations
        captured_activations: List[Tensor] = []

        def capture_hook(module, args):
            """Hook to capture activations without modifying them."""
            hidden_states = args[0]  # Shape: [batch, seq_len, hidden_dim]
            captured_activations.append(hidden_states.detach().clone())
            return args  # Return unchanged

        # Tokenize and run forward pass with hook
        input_ids = backend.tokenize(prompt)

        with backend.hooks_context([(layer, capture_hook)]):
            _ = backend.get_logits(input_ids)

        # Get activations: shape [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
        activations = captured_activations[0].squeeze(0)

        # Compute dot products for each token
        dot_products = torch.matmul(activations, vec)  # [seq_len]

        # Aggregate
        if self._aggregation == "mean":
            score = dot_products.mean().item()
        elif self._aggregation == "max":
            score = dot_products.max().item()
        elif self._aggregation == "last":
            score = dot_products[-1].item()
        else:
            raise ValueError(f"Unknown aggregation: {self._aggregation}")

        return ClassificationResult(
            score=score,
            prompt=prompt,
            layer=layer,
            metadata={
                "aggregation": self._aggregation,
                "normalized": self._normalize_vector,
                "num_tokens": len(dot_products),
            },
        )
