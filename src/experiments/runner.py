"""Experiment runner for steering vector classification experiments."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from steerex import HuggingFaceBackend, ModelBackend

from src.classifiers.base import BaseClassifier
from src.classifiers.dot_product import DotProductClassifier
from src.classifiers.logit_shift import LogitShiftClassifier
from src.experiments.config import ClassifierConfig, ExperimentConfig
from src.experiments.metrics import (
    LabeledMisalignmentSample,
    LabeledSycophancySample,
    compute_roc_auc,
    get_label_statistics,
    get_sycophancy_label_statistics,
    label_misalignment_dataset,
)
from src.evals.misalignment import load_misalignment_dataset
from src.utils import (
    apply_chat_template,
    format_alpaca_prompt,
    load_alpaca_dataset,
    load_model,
    load_vector,
)


@dataclass
class ClassifierResults:
    """Results for a single classifier on a single layer."""

    classifier_name: str
    layer: int

    # Misalignment dataset results
    misalignment_scores: List[float]
    misalignment_labels: List[int]
    roc_auc: float

    # Control dataset results
    control_scores: List[float]
    control_mean: float
    control_std: float

    # ROC curve data
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


@dataclass
class ExperimentResults:
    """Complete experiment results."""

    config: ExperimentConfig
    timestamp: str

    # Labeling statistics
    label_stats: dict

    # Results per classifier per layer: classifier_name -> layer -> results
    results: Dict[str, Dict[int, ClassifierResults]] = field(default_factory=dict)

    # Best configuration
    best_classifier: str = ""
    best_layer: int = 0
    best_roc_auc: float = 0.0


class ExperimentRunner:
    """Runs classification experiments with steering vectors."""

    def __init__(
        self,
        config: ExperimentConfig,
        backend: ModelBackend | None = None,
        tokenizer=None,
        steering_vector: torch.Tensor | None = None,
        eval_samples: List[LabeledMisalignmentSample] | List[LabeledSycophancySample] | None = None,
        control_prompts: List[str] | None = None,
    ):
        """
        Initialize ExperimentRunner.

        Args:
            config: Experiment configuration.
            backend: Optional pre-loaded backend (useful for notebook reuse).
            tokenizer: Optional pre-loaded tokenizer.
            steering_vector: Optional pre-loaded steering vector.
            eval_samples: Optional pre-labeled evaluation samples. If provided,
                skips loading from DatasetConfig.
            control_prompts: Optional pre-formatted control prompts. If provided,
                skips loading Alpaca dataset.
        """
        self.config = config
        self.backend = backend
        self.tokenizer = tokenizer
        self.steering_vector = steering_vector
        self.eval_samples = eval_samples
        self.control_prompts = control_prompts

    def setup(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
    ) -> None:
        """
        Load model and steering vector if not already provided.

        Args:
            model_name: HuggingFace model name.
            dtype: Model dtype.
            device_map: Device mapping strategy.
        """
        if self.backend is None:
            model, tokenizer = load_model(
                model_name,
                dtype=dtype,
                device_map=device_map,
            )
            self.tokenizer = tokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.backend = HuggingFaceBackend(model, tokenizer, device)

        if self.steering_vector is None and self.config.vector_path:
            self.steering_vector, _ = load_vector(self.config.vector_path)

    def _create_classifier(self, config: ClassifierConfig) -> BaseClassifier:
        """Factory method to create classifiers from config."""
        if config.type == "dot_product":
            return DotProductClassifier(
                aggregation=config.aggregation,
                similarity=config.similarity,
                center=config.center,
                scale=config.scale,
                epsilon=config.epsilon,
            )
        elif config.type == "logit_shift":
            return LogitShiftClassifier(
                strength=config.strength,
                use_symmetric_kl=config.use_symmetric_kl,
            )
        else:
            raise ValueError(f"Unknown classifier type: {config.type}")

    def _build_classification_prompt(
        self,
        sample: LabeledMisalignmentSample | LabeledSycophancySample,
    ) -> str:
        """
        Build the classification prompt, optionally including the model response.

        Args:
            sample: Labeled sample with messages and model_response.

        Returns:
            Formatted prompt string ready for classification.
        """
        if not self.config.include_response:
            # Use pre-computed prompt (without response)
            return sample.prompt

        # Build messages with response included
        messages_with_response = sample.messages + [
            {"role": "assistant", "content": sample.model_response}
        ]

        # Apply chat template (without generation prompt since response is included)
        return self.tokenizer.apply_chat_template(
            messages_with_response,
            tokenize=False,
            add_generation_prompt=False,
        )

    def run(self, verbose: bool = True) -> ExperimentResults:
        """
        Run the full experiment.

        Args:
            verbose: Show progress bars.

        Returns:
            ExperimentResults with all metrics.
        """
        if self.backend is None:
            raise RuntimeError("Backend not initialized. Call setup() first.")
        if self.steering_vector is None:
            raise RuntimeError("Steering vector not provided.")

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Load datasets (or use pre-split samples to prevent data leakage)
        if self.eval_samples is not None:
            # Use pre-split samples directly
            labeled_samples = self.eval_samples
            # Detect sample type and use appropriate statistics function
            is_sycophancy = (
                len(labeled_samples) > 0
                and isinstance(labeled_samples[0], LabeledSycophancySample)
            )
            if is_sycophancy:
                label_stats = get_sycophancy_label_statistics(labeled_samples)
                if verbose:
                    print("Step 1: Using pre-split sycophancy evaluation samples...")
                    print(f"  {label_stats['total']} samples provided")
                    print(f"  Sycophancy rate: {label_stats['sycophancy_rate']:.1%}")
            else:
                label_stats = get_label_statistics(labeled_samples)
                if verbose:
                    print("Step 1: Using pre-split evaluation samples...")
                    print(f"  {label_stats['total']} samples provided")
                    print(f"  Misalignment rate: {label_stats['misalignment_rate']:.1%}")
        else:
            # Load and label from DatasetConfig
            misalignment_dataset = load_misalignment_dataset(
                split=self.config.dataset.misalignment_split
            )
            if verbose:
                print("Step 1: Labeling misalignment dataset...")
            labeled_samples = label_misalignment_dataset(
                backend=self.backend,
                tokenizer=self.tokenizer,
                dataset=misalignment_dataset,
                n_samples=self.config.dataset.n_misalignment_samples,
                seed=self.config.dataset.seed,
                verbose=verbose,
            )
            label_stats = get_label_statistics(labeled_samples)
            if verbose:
                print(f"  Labeled {label_stats['total']} samples")
                print(f"  Misalignment rate: {label_stats['misalignment_rate']:.1%}")

        # Prepare control prompts (or use pre-split)
        if self.control_prompts is not None:
            control_prompts = self.control_prompts
            if verbose:
                print(f"  Using {len(control_prompts)} pre-split control prompts")
        else:
            control_dataset = load_alpaca_dataset(
                n_samples=self.config.dataset.n_control_samples,
                seed=self.config.dataset.seed,
            )
            control_prompts = [
                apply_chat_template(self.tokenizer, format_alpaca_prompt(row))
                for row in control_dataset
            ]

        # Step 2: Run classifiers
        all_results: Dict[str, Dict[int, ClassifierResults]] = {}

        if verbose:
            response_mode = "with response" if self.config.include_response else "prompt only"
            print(f"  Classification mode: {response_mode}")

        for classifier_config in self.config.classifiers:
            classifier = self._create_classifier(classifier_config)
            if verbose:
                print(f"\nStep 2: Running {classifier.name}...")

            all_results[classifier.name] = {}

            for layer in self.config.eval_layers:
                if verbose:
                    print(f"  Layer {layer}...")

                # Classify misalignment samples
                misalignment_scores = []
                samples_iter = (
                    tqdm(labeled_samples, desc="    Classifying")
                    if verbose and tqdm
                    else labeled_samples
                )
                for sample in samples_iter:
                    prompt = self._build_classification_prompt(sample)
                    result = classifier.classify(
                        backend=self.backend,
                        prompt=prompt,
                        steering_vector=self.steering_vector,
                        layer=layer,
                    )
                    misalignment_scores.append(result.score)

                labels = [s.label for s in labeled_samples]

                # Compute ROC-AUC
                auc, fpr, tpr, thresholds = compute_roc_auc(misalignment_scores, labels)

                # Classify control samples
                control_scores = []
                control_iter = (
                    tqdm(control_prompts, desc="    Control")
                    if verbose and tqdm
                    else control_prompts
                )
                for prompt in control_iter:
                    result = classifier.classify(
                        backend=self.backend,
                        prompt=prompt,
                        steering_vector=self.steering_vector,
                        layer=layer,
                    )
                    control_scores.append(result.score)

                all_results[classifier.name][layer] = ClassifierResults(
                    classifier_name=classifier.name,
                    layer=layer,
                    misalignment_scores=misalignment_scores,
                    misalignment_labels=labels,
                    roc_auc=auc,
                    control_scores=control_scores,
                    control_mean=float(np.mean(control_scores)),
                    control_std=float(np.std(control_scores)),
                    fpr=fpr,
                    tpr=tpr,
                    thresholds=thresholds,
                )

                if verbose:
                    print(f"    ROC-AUC: {auc:.4f}")
                    print(
                        f"    Control: {np.mean(control_scores):.4f} +/- {np.std(control_scores):.4f}"
                    )

        # Find best configuration
        best_auc = 0.0
        best_classifier = ""
        best_layer = 0

        for clf_name, layer_results in all_results.items():
            for layer, results in layer_results.items():
                if results.roc_auc > best_auc:
                    best_auc = results.roc_auc
                    best_classifier = clf_name
                    best_layer = layer

        return ExperimentResults(
            config=self.config,
            timestamp=datetime.now().isoformat(),
            label_stats=label_stats,
            results=all_results,
            best_classifier=best_classifier,
            best_layer=best_layer,
            best_roc_auc=best_auc,
        )

    def save_results(
        self, results: ExperimentResults, path: Path | None = None
    ) -> Path:
        """
        Save results to JSON.

        Args:
            results: Experiment results.
            path: Output path (default: config.output_dir/<name>_<timestamp>.json).

        Returns:
            Path to saved file.
        """
        if path is None:
            timestamp = results.timestamp.replace(":", "-").replace(".", "-")
            path = self.config.output_dir / f"{self.config.name}_{timestamp}.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to JSON-serializable format
        data = {
            "config": self.config.model_dump(mode="json"),
            "timestamp": results.timestamp,
            "label_stats": results.label_stats,
            "best_classifier": results.best_classifier,
            "best_layer": results.best_layer,
            "best_roc_auc": results.best_roc_auc,
            "results": {},
        }

        for clf_name, layer_results in results.results.items():
            data["results"][clf_name] = {}
            for layer, r in layer_results.items():
                data["results"][clf_name][str(layer)] = {
                    "roc_auc": r.roc_auc,
                    "control_mean": r.control_mean,
                    "control_std": r.control_std,
                    "n_samples": len(r.misalignment_scores),
                }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path
