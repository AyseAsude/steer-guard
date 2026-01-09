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
    3. Compute similarity between steering vector and each token's activation
    4. Aggregate scores (mean/max/last) as classification score

    Higher score -> model's activations align more with steering direction.
    """

    def __init__(
        self,
        aggregation: Literal["mean", "max", "last"] = "mean",
        similarity: Literal["dot", "cosine"] = "dot",
    ):
        """
        Initialize DotProductClassifier.

        Args:
            aggregation: How to aggregate per-token scores.
                - "mean": Average across all tokens
                - "max": Maximum score
                - "last": Use only the last token
            similarity: Similarity metric to use.
                - "dot": Standard dot product (x · v), with normalized v
                - "cosine": Cosine similarity (x · v) / (||x|| ||v||)
        """
        self._aggregation = aggregation
        self._similarity = similarity

    @property
    def name(self) -> str:
        return f"DotProduct({self._similarity}, {self._aggregation})"

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

        # Compute similarity scores for each token
        scores = self._compute_similarity(activations, vec)

        # Aggregate
        if self._aggregation == "mean":
            score = scores.mean().item()
        elif self._aggregation == "max":
            score = scores.max().item()
        elif self._aggregation == "last":
            score = scores[-1].item()
        else:
            raise ValueError(f"Unknown aggregation: {self._aggregation}")

        return ClassificationResult(
            score=score,
            prompt=prompt,
            layer=layer,
            metadata={
                "aggregation": self._aggregation,
                "similarity": self._similarity,
                "num_tokens": len(scores),
            },
        )

    def _compute_similarity(self, activations: Tensor, vec: Tensor) -> Tensor:
        """
        Compute similarity between activations and steering vector.

        Args:
            activations: Token activations [seq_len, hidden_dim]
            vec: Steering vector [hidden_dim]

        Returns:
            Similarity scores [seq_len]
        """
        if self._similarity == "dot":
            # Standard dot product with normalized vector
            vec = vec / vec.norm()
            return torch.matmul(activations, vec)
        elif self._similarity == "cosine":
            # Cosine similarity: (x · v) / (||x|| ||v||)
            vec_norm = vec / vec.norm()
            act_norms = activations.norm(dim=1, keepdim=True)
            activations_norm = activations / act_norms
            return torch.matmul(activations_norm, vec_norm)
        else:
            raise ValueError(f"Unknown similarity: {self._similarity}")
