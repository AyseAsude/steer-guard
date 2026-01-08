"""Logit-shift (KL divergence) based classifier."""

import torch
from torch import Tensor

from steering_vectors import ModelBackend, VectorSteering

from src.classifiers.base import BaseClassifier, ClassificationResult


class LogitShiftClassifier(BaseClassifier):
    """
    Classifier using KL divergence between steered and unsteered logits.

    Algorithm:
    1. Forward pass WITHOUT steering -> get logits at last token position
    2. Forward pass WITH steering (add vector at layer L) -> get steered logits
    3. Compute KL divergence: KL(unsteered || steered)

    Higher divergence -> steering has larger effect on this prompt.
    """

    def __init__(
        self,
        strength: float = 1.0,
        use_symmetric_kl: bool = False,
    ):
        """
        Initialize LogitShiftClassifier.

        Args:
            strength: Steering strength multiplier.
            use_symmetric_kl: If True, use (KL(P||Q) + KL(Q||P))/2.
        """
        self._strength = strength
        self._use_symmetric_kl = use_symmetric_kl

    @property
    def name(self) -> str:
        sym = "_sym" if self._use_symmetric_kl else ""
        return f"LogitShift(s={self._strength}{sym})"

    def classify(
        self,
        backend: ModelBackend,
        prompt: str,
        steering_vector: Tensor,
        layer: int,
    ) -> ClassificationResult:
        """
        Compute KL divergence classification score.

        Args:
            backend: Model backend.
            prompt: Formatted prompt string.
            steering_vector: Steering vector (hidden_dim,).
            layer: Layer to apply steering at.

        Returns:
            ClassificationResult with KL divergence score.
        """
        input_ids = backend.tokenize(prompt)

        # Forward pass WITHOUT steering
        unsteered_logits = backend.get_logits(input_ids, hooks=[])
        # Get last token logits: [vocab_size]
        unsteered_last = unsteered_logits[0, -1, :].float()

        # Create steering hook using VectorSteering
        steering = VectorSteering(vector=steering_vector)
        hook = steering.create_hook(token_slice=None, strength=self._strength)

        # Forward pass WITH steering
        steered_logits = backend.get_logits(input_ids, hooks=[(layer, hook)])
        steered_last = steered_logits[0, -1, :].float()

        # Compute KL divergence
        kl_div = self._compute_kl(unsteered_last, steered_last)

        if self._use_symmetric_kl:
            kl_reverse = self._compute_kl(steered_last, unsteered_last)
            score = (kl_div + kl_reverse) / 2
        else:
            score = kl_div

        return ClassificationResult(
            score=score,
            prompt=prompt,
            layer=layer,
            metadata={
                "strength": self._strength,
                "symmetric_kl": self._use_symmetric_kl,
            },
        )

    def _compute_kl(self, logits_p: Tensor, logits_q: Tensor) -> float:
        """Compute KL(P || Q) from logits."""
        log_p = torch.log_softmax(logits_p, dim=-1)
        log_q = torch.log_softmax(logits_q, dim=-1)
        p = torch.softmax(logits_p, dim=-1)

        kl = (p * (log_p - log_q)).sum().item()
        return kl
