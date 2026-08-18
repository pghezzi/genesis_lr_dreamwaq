"""Train a standardized-feature PCA/prototype RBF classifier and Bayes filter."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    build_filter_from_search_result, search_bayes_filter_hyperparameters,
    search_prototype_rbf_hyperparameters_dataloader,
)
from .util_func import (
    classifier_metrics_from_scores, collect_engineered_scores, evaluate_bayes_from_scores,
    extract_dataset_features, fit_standardizer,
    json_safe, load_training_files, make_terrain_extractor, save_results, sequence_ids_for,
    transition_training_kwargs, processing_batch_size,
)


class TerrainFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: Sequence[Any]) -> None:
        self.features, self.labels = torch.as_tensor(features).float(), list(labels)
        if self.features.ndim != 2 or self.features.shape[0] != len(self.labels):
            raise ValueError("features must be [N,D] with one label per row")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        return self.features[index], self.labels[index]


def _loader(features: torch.Tensor, labels: Sequence[Any], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TerrainFeatureDataset(features, labels), batch_size=batch_size, shuffle=shuffle)


def main() -> None:
    args, files = load_training_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output_dir or (
        Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
        f"rbf_prototype_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output.mkdir(parents=True, exist_ok=True)
    extractor = make_terrain_extractor(files["calibration"])
    train = torch.load(files["train"], map_location="cpu", weights_only=False)
    validation = torch.load(files["validation"], map_location="cpu", weights_only=False)
    train_batch = processing_batch_size(args, len(train["labels"]))
    validation_batch = processing_batch_size(args, len(validation["labels"]))
    train_features = extract_dataset_features(extractor, train, chunk_size=train_batch)
    validation_features = extract_dataset_features(extractor, validation, chunk_size=validation_batch)
    standardizer = fit_standardizer(train_features, train["labels"], batch_size=train_batch)
    train_features = standardizer.transform(train_features)
    validation_features = standardizer.transform(validation_features)

    search_start = time.perf_counter()
    classifier, search_results = search_prototype_rbf_hyperparameters_dataloader(
        lambda: _loader(train_features, train["labels"], train_batch, True),
        lambda: _loader(validation_features, validation["labels"], validation_batch, False),
        base_config={
            "feature_dim": train_features.shape[1], "device": device,
            "dtype": torch.float32, "kmeans_fit_mode": "mini_batch",
            "store_training_data": False, "random_seed": 0,
        },
        search_space={
            "pca_dim": [16, 24, 32], "prototypes_per_class": [4, 8, 12],
            "gamma": ["scale", 0.25, 0.5, 1.0],
            "metric_type": ["euclidean", "diag_mahalanobis"],
            "aggregation": ["logsumexp"], "prototype_init": ["kmeans++"],
            "prototype_epochs": [5, 10], "initialization_sample_size": [1024, 2048],
            "reset_counts_each_epoch": [True], "variance_shrinkage": [0.1, 0.3],
        },
    )
    search_runtime = time.perf_counter() - search_start
    structural_test = torch.load(files["structural_test"], map_location="cpu", weights_only=False)
    test_scores = collect_engineered_scores(
        classifier, extractor, standardizer, structural_test,
        chunk_size=processing_batch_size(args, len(structural_test["labels"])),
    )
    instantaneous = classifier_metrics_from_scores(classifier, test_scores, structural_test["labels"])
    del train, validation, train_features, validation_features
    del structural_test, test_scores

    calibration = torch.load(files["calibration"], map_location="cpu", weights_only=False)
    calibration_labels = classifier._normalize_labels(calibration["labels"])
    calibration_scores = collect_engineered_scores(
        classifier, extractor, standardizer, calibration,
        chunk_size=processing_batch_size(args, len(calibration_labels)),
    )
    del calibration
    bayes_validation = torch.load(files["bayes_validation"], map_location="cpu", weights_only=False)
    bayes_labels = classifier._normalize_labels(bayes_validation["labels"])
    bayes_ids = sequence_ids_for(bayes_validation)
    bayes_scores = collect_engineered_scores(
        classifier, extractor, standardizer, bayes_validation,
        chunk_size=processing_batch_size(args, len(bayes_labels)),
    )
    del bayes_validation
    prior = {label: 1 / len(classifier.class_ids) for label in classifier.class_ids}
    transition_kwargs = transition_training_kwargs(files)
    classifier.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bayes_start = time.perf_counter()
    filter_results = search_bayes_filter_hyperparameters(
        classifier, None, bayes_labels, prior,
        sequence_ids=bayes_ids, filter_scores=bayes_scores,
        observation_calibration_scores=calibration_scores,
        observation_calibration_labels=calibration_labels,
        temperatures=[0.5, 0.75, 1.0, 1.5, 2.0], stay_probabilities=[0.90, 0.94, 0.97],
        transition_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        evidence_powers=[0.50, 0.75, 1.0], min_evidence_powers=[0.10, 0.25],
        confidence_gammas=[1.0, 2.0], observation_modes=["soft"],
        observation_pseudocounts=[0.5], scoring="balanced_accuracy", device="cpu",
        **transition_kwargs,
    )
    bayes_runtime = time.perf_counter() - bayes_start
    del bayes_scores, calibration_scores, transition_kwargs
    selected = filter_results[0]
    bayes_filter, temperature = build_filter_from_search_result(classifier, selected, prior, device=device)
    classifier.temperature = temperature
    ordered_test = torch.load(files["ordered_test"], map_location="cpu", weights_only=False)
    ordered_scores = collect_engineered_scores(
        classifier, extractor, standardizer, ordered_test,
        chunk_size=processing_batch_size(args, len(ordered_test["labels"])),
    )
    ordered_instantaneous = classifier_metrics_from_scores(
        classifier, ordered_scores, ordered_test["labels"], temperature
    )
    bayesian = evaluate_bayes_from_scores(
        classifier, bayes_filter, ordered_scores, ordered_test["labels"],
        sequence_ids_for(ordered_test), temperature,
    )

    extractor.save(output / "extractor.pt")
    standardizer.save(output / "standardizer.pt")
    classifier.save(output / "classifier.pt")
    bayes_filter.save(output / "bayes_filter.pt")
    torch.save(search_results, output / "classifier_search.pt")
    torch.save(filter_results, output / "bayes_search.pt")
    results = {
        "schema_version": 1, "method": "RBF Prototype", "require_feature": True,
        "best_hyperparameters": search_results[0].params,
        "validation_score": search_results[0].validation_accuracy,
        "instantaneous": instantaneous,
        "model": {"prototype_count": search_results[0].num_prototypes,
                  "model_size_bytes": (output / "classifier.pt").stat().st_size},
        "runtime_seconds": {"search_or_training": search_runtime, "bayes_search": bayes_runtime},
        "bayes": {
            "selected_temperature": temperature,
            "filter_parameters": {k: selected[k] for k in (
                "stay_probability", "evidence_power", "min_evidence_power", "confidence_gamma",
                "transition_alpha", "transition_matrix", "transition_source",
                "observation_mode", "observation_pseudocount")},
            "validation_score": selected["objective"],
            "unfiltered_metrics": ordered_instantaneous, "metrics": bayesian,
        }, "data_files": files,
    }
    save_results(output / "results.json", results)
    save_results(output / "params.json", {
        "classifier": results["best_hyperparameters"], "temperature": temperature,
        "bayes_filter": results["bayes"]["filter_parameters"],
    })
    save_results(output / "metrics.json", {"instantaneous": instantaneous,
                                            "ordered_instantaneous": ordered_instantaneous,
                                            "bayesian": bayesian})
    torch.save(json_safe(results), output / "results.pt")
    print(f"Saved RBF Prototype run to {output}")


if __name__ == "__main__":
    main()
