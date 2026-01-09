"""Experiment configuration dataclasses."""

from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field


class ClassifierConfig(BaseModel):
    """Configuration for a classifier.

    Fields are split by classifier type - only relevant fields are used.
    """

    type: Literal["dot_product", "logit_shift"] = "dot_product"

    # DotProductClassifier options (used when type="dot_product")
    aggregation: Literal["mean", "max", "last"] = "mean"
    similarity: Literal["dot", "cosine"] = "dot"

    # LogitShiftClassifier options (used when type="logit_shift")
    strength: float = 1.0
    use_symmetric_kl: bool = False


class DatasetConfig(BaseModel):
    """Configuration for datasets."""

    # Misalignment dataset
    misalignment_split: str | None = None  # None = all, or "article_questions"/"textbook_questions"
    n_misalignment_samples: int | None = None  # None = all

    # Control dataset (Alpaca)
    n_control_samples: int = 100

    # Reproducibility
    seed: int = 42


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    # Experiment identity
    name: str = "steering_classifier_experiment"

    # Steering vector
    vector_path: Path | None = None  # Path to saved vector (.pt file)

    # Layers to evaluate
    eval_layers: List[int] = Field(default_factory=lambda: [10])

    # Classifiers to run
    classifiers: List[ClassifierConfig] = Field(
        default_factory=lambda: [ClassifierConfig()]
    )

    # Dataset configuration
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Output
    output_dir: Path = Field(default=Path("./results"))

    class Config:
        arbitrary_types_allowed = True
