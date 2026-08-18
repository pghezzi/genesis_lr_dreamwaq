"""Shared helpers for terrain-classifier training scripts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    FeatureStandardizer, collect_classifier_scores_batched, evaluate_predictions,
    fit_nn, run_filter_sequences,
)


def make_terrain_extractor(calibration_file: str | Path) -> SobelDepthTerrainFeatureExtractor:
    calibration = torch.load(calibration_file, map_location="cpu", weights_only=False)
    images = calibration["depth_images"].float()
    extractor = SobelDepthTerrainFeatureExtractor(
        output_size=tuple(images.shape[-2:]), min_depth=0.02, max_depth=1.0,
        far_depth=0.6, close_depth=0.15, close_residual_threshold=0.05,
        sobel_edge_threshold=0.007, depth_scale=None,
    )
    extractor.fit_reference_model(images, calibration["orientation_rpy"])
    return extractor


def extract_in_chunks(extractor: Any, depth_images: torch.Tensor, orientation_rpy: torch.Tensor,
                      angular_velocity: torch.Tensor, chunk_size: int = 256,
                      device: str | torch.device = "cpu") -> torch.Tensor:
    outputs = []
    for start in range(0, depth_images.shape[0], chunk_size):
        stop = min(start + chunk_size, depth_images.shape[0])
        with torch.inference_mode():
            features = extractor.extract_batch(
                depth_images[start:stop].float(), orientation_rpy[start:stop], angular_velocity[start:stop]
            )
        outputs.append(features.detach().to(device))
    if not outputs:
        raise ValueError("cannot extract features from an empty dataset")
    return torch.cat(outputs)


def extract_dataset_features(extractor: Any, data: Mapping[str, Any], chunk_size: int = 256) -> torch.Tensor:
    return extract_in_chunks(extractor, data["depth_images"], data["orientation_rpy"],
                             data["angular_velocity"], chunk_size=chunk_size)


def fit_standardizer(features: torch.Tensor, labels: Sequence[Any], batch_size: int = 512) -> FeatureStandardizer:
    """Fit strictly from supplied structural-training feature rows."""
    loader = DataLoader(TensorDataset(features.cpu(), torch.arange(len(labels))), batch_size=batch_size)
    return FeatureStandardizer().fit(loader)


def collect_engineered_scores(
    classifier: Any, extractor: Any, standardizer: FeatureStandardizer,
    data: Mapping[str, Any], chunk_size: int = 256,
) -> torch.Tensor:
    """Extract, standardize, and score chunks without materializing all features."""
    def batches():
        n = data["depth_images"].shape[0]
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            with torch.inference_mode():
                features = extractor.extract_batch(
                    data["depth_images"][start:stop].float(),
                    data["orientation_rpy"][start:stop],
                    data["angular_velocity"][start:stop],
                )
            yield standardizer.transform(features)

    return collect_classifier_scores_batched(classifier, batches(), cache_device="cpu")


def collect_raw_depth_scores(
    classifier: Any, depth_images: torch.Tensor, chunk_size: int = 256,
) -> torch.Tensor:
    """Score raw depth chunks without creating a full float/channel-expanded copy."""
    def batches():
        for start in range(0, depth_images.shape[0], chunk_size):
            stop = min(start + chunk_size, depth_images.shape[0])
            yield depth_images[start:stop].unsqueeze(1).float()

    return collect_classifier_scores_batched(classifier, batches(), cache_device="cpu")


def classifier_metrics(classifier: Any, inputs: Any, labels: Sequence[Any], temperature: float = 1.0) -> dict[str, Any]:
    scores = classifier.decision_function(inputs).detach().cpu()
    return classifier_metrics_from_scores(classifier, scores, labels, temperature)


def classifier_metrics_from_scores(
    classifier: Any, scores: torch.Tensor, labels: Sequence[Any], temperature: float = 1.0,
) -> dict[str, Any]:
    ordered_labels = list(classifier.class_ids)
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / temperature, dim=1)
    truth = list(labels.detach().cpu().tolist() if torch.is_tensor(labels) else labels)
    predicted = [ordered_labels[i] for i in probabilities.argmax(1).tolist()]
    metrics = evaluate_predictions(truth, predicted, ordered_labels).as_dict()
    encoded = torch.tensor([classifier.class_to_index[label] for label in truth], dtype=torch.long)
    targets = F.one_hot(encoded, len(ordered_labels)).to(probabilities.dtype)
    metrics["nll"] = float(F.nll_loss(probabilities.clamp_min(classifier.eps).log(), encoded))
    metrics["brier"] = float((probabilities - targets).square().sum(1).mean())
    return metrics


def evaluate_bayes(classifier: Any, bayes_filter: Any, inputs: Any, labels: Sequence[Any],
                   sequence_ids: Sequence[Any], temperature: float) -> dict[str, Any]:
    scores = classifier.decision_function(inputs).detach().cpu()
    return evaluate_bayes_from_scores(
        classifier, bayes_filter, scores, labels, sequence_ids, temperature
    )


def evaluate_bayes_from_scores(
    classifier: Any, bayes_filter: Any, scores: torch.Tensor, labels: Sequence[Any],
    sequence_ids: Sequence[Any], temperature: float,
) -> dict[str, Any]:
    probabilities = F.softmax(torch.as_tensor(scores).cpu() / temperature, dim=1).clamp_min(1e-8)
    probabilities = probabilities / probabilities.sum(1, keepdim=True)
    ordered_labels = list(classifier.class_ids)
    predictions, _, _ = run_filter_sequences(
        bayes_filter, probabilities, sequence_ids=sequence_ids, return_traces=False
    )
    truth = classifier._normalize_labels(labels)
    return evaluate_predictions(truth, predictions, ordered_labels, sequence_ids=sequence_ids).as_dict()


def sequence_ids_for(data: Mapping[str, Any]) -> torch.Tensor:
    if "sequence_ids" in data:
        return torch.as_tensor(data["sequence_ids"])
    if "episode_ids" in data:
        return torch.as_tensor(data["episode_ids"])
    per_episode = int(data.get("per_eps", 0))
    n = len(data["labels"])
    if per_episode <= 0:
        return torch.zeros(n, dtype=torch.long)
    return torch.arange(math.ceil(n / per_episode)).repeat_interleave(per_episode)[:n]


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(path: str | Path, results: Mapping[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(json_safe(results), stream, indent=2, sort_keys=True)


def load_training_files(argv: Optional[Sequence[str]] = None) -> tuple[argparse.Namespace, dict[str, Path]]:
    parser = argparse.ArgumentParser(description="Train a terrain classifier")
    parser.add_argument("--classifier_folder", required=True, type=Path)
    parser.add_argument("--bayesian_folder", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch/chunk size for training, extraction, and classifier scoring")
    parser.add_argument("--no_batch_processing", action="store_true",
                        help="use each complete split as one batch (may require substantial RAM/VRAM)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    files = {
        "train": args.classifier_folder / "train.pt",
        "validation": args.classifier_folder / "val.pt",
        "calibration": args.classifier_folder / "calibration.pt",
        "structural_test": args.classifier_folder / "test.pt",
        "bayes_validation": args.bayesian_folder / "val.pt",
        "ordered_test": args.bayesian_folder / "test.pt",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required data files: " + ", ".join(missing))
    # Optional ordered transition-training data. Prefer train over calibration;
    # when neither exists the search explicitly warns before reusing validation.
    for name in ("train", "calibration"):
        candidate = args.bayesian_folder / f"{name}.pt"
        if candidate.is_file():
            files["transition_training"] = candidate
            break
    return args, files


def processing_batch_size(args: argparse.Namespace, sample_count: int) -> int:
    """Resolve the shared bounded-batch/full-split processing choice."""
    if sample_count <= 0:
        raise ValueError("dataset split is empty")
    return sample_count if args.no_batch_processing else args.batch_size


def transition_training_kwargs(files: Mapping[str, Path]) -> dict[str, Any]:
    """Load optional classifier-independent ordered transition-training truth."""
    path = files.get("transition_training")
    if path is None:
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "transition_training_labels": data["labels"],
        "transition_training_sequence_ids": sequence_ids_for(data),
    }


def get_files_for_training() -> tuple[Path, ...]:
    """Compatibility wrapper returning paths in the historical order."""
    _, files = load_training_files()
    return tuple(files[key] for key in (
        "train", "validation", "calibration", "structural_test", "bayes_validation", "ordered_test"
    ))


__all__ = [
    "classifier_metrics", "classifier_metrics_from_scores", "collect_engineered_scores",
    "collect_raw_depth_scores", "evaluate_bayes", "evaluate_bayes_from_scores",
    "extract_dataset_features", "extract_in_chunks",
    "FeatureStandardizer", "fit_nn", "fit_standardizer", "json_safe", "load_training_files",
    "make_terrain_extractor", "save_results", "sequence_ids_for",
    "transition_training_kwargs", "processing_batch_size",
]
