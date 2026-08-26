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
    search_prototype_rbf_hyperparameters_dataloader,
)
from .util_func import (
    classifier_metrics_from_scores, collect_engineered_scores,
    extract_dataset_features, fit_standardizer,
    json_safe, load_training_files, make_terrain_extractor, save_results, sequence_ids_for,
    processing_batch_size, run_staged_sequential_pipeline,
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
    base_config = {
        "feature_dim": train_features.shape[1], "device": device,
        "dtype": torch.float32, "kmeans_fit_mode": "mini_batch",
        "store_training_data": False, "random_seed": 0,
    }
    stage1_classifier, stage1_results = search_prototype_rbf_hyperparameters_dataloader(
        lambda: _loader(train_features, train["labels"], train_batch, True),
        lambda: _loader(validation_features, validation["labels"], validation_batch, False),
        base_config=base_config,
        search_space={
            "pca_dim": [8, 12, 16], "prototypes_per_class": [12, 16, 24, 32],
            "gamma": [0.75, 1.0, 1.5, 2.0, 4.0], "metric_type": ["euclidean"],
            "aggregation": ["logsumexp"], "prototype_init": ["kmeans++"],
            "prototype_epochs": [10], "initialization_sample_size": [2048],
            "reset_counts_each_epoch": [True],
        },
    )
    stage2_results, stage2_candidates = [], []
    for seed in stage1_results[:3]:
        fixed = dict(seed.params)
        for key in ("prototype_epochs", "initialization_sample_size"):
            fixed.pop(key, None)
        candidate, trials = search_prototype_rbf_hyperparameters_dataloader(
            lambda: _loader(train_features, train["labels"], train_batch, True),
            lambda: _loader(validation_features, validation["labels"], validation_batch, False),
            base_config={**base_config, **fixed},
            search_space={"prototype_epochs": [10, 20],
                          "initialization_sample_size": [2048, 4096]},
        )
        merged = [{"params": {**fixed, **r.params}, "validation_accuracy": r.validation_accuracy,
                   "validation_nll": r.validation_nll, "validation_brier": r.validation_brier,
                   "num_prototypes": r.num_prototypes} for r in trials]
        stage2_results.extend(merged)
        best = min(merged, key=lambda r: (-r["validation_accuracy"], r["validation_nll"], r["validation_brier"]))
        stage2_candidates.append((best, candidate))
    stage2_results.sort(key=lambda r: (-r["validation_accuracy"], r["validation_nll"], r["validation_brier"]))
    stage1_best = {"params": stage1_results[0].params,
                   "validation_accuracy": stage1_results[0].validation_accuracy,
                   "validation_nll": stage1_results[0].validation_nll,
                   "validation_brier": stage1_results[0].validation_brier,
                   "num_prototypes": stage1_results[0].num_prototypes}
    stage2_best, stage2_classifier = min(
        stage2_candidates, key=lambda item: (-item[0]["validation_accuracy"], item[0]["validation_nll"], item[0]["validation_brier"]))
    selected_stage, selected_record, classifier = min(
        [("stage1", stage1_best, stage1_classifier), ("stage2", stage2_best, stage2_classifier)],
        key=lambda item: (-item[1]["validation_accuracy"], item[1]["validation_nll"], item[1]["validation_brier"]),
    )
    search_runtime = time.perf_counter() - search_start
    structural_test = torch.load(files["structural_test"], map_location="cpu", weights_only=False)
    structural_batch = processing_batch_size(args, len(structural_test["labels"]))
    stage1_test = classifier_metrics_from_scores(stage1_classifier, collect_engineered_scores(
        stage1_classifier, extractor, standardizer, structural_test, chunk_size=structural_batch), structural_test["labels"])
    stage2_test = classifier_metrics_from_scores(stage2_classifier, collect_engineered_scores(
        stage2_classifier, extractor, standardizer, structural_test, chunk_size=structural_batch), structural_test["labels"])
    instantaneous = stage1_test if selected_stage == "stage1" else stage2_test
    del train, validation, train_features, validation_features
    del structural_test

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
    classifier.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bayes_start = time.perf_counter()
    ordered_test = torch.load(files["ordered_test"], map_location="cpu", weights_only=False)
    ordered_scores = collect_engineered_scores(
        classifier, extractor, standardizer, ordered_test,
        chunk_size=processing_batch_size(args, len(ordered_test["labels"])),
    )
    sequential_search, filter_results, legacy_bayes = run_staged_sequential_pipeline(
        classifier, calibration_scores, calibration_labels, bayes_scores, bayes_labels, bayes_ids,
        ordered_scores, ordered_test["labels"], sequence_ids_for(ordered_test), output,
    )
    bayes_runtime = time.perf_counter() - bayes_start
    temperature = legacy_bayes["selected_temperature"]
    classifier.temperature = temperature
    ordered_instantaneous = legacy_bayes["unfiltered_metrics"]
    bayesian = legacy_bayes["metrics"]

    extractor.save(output / "extractor.pt")
    standardizer.save(output / "standardizer.pt")
    classifier.save(output / "classifier.pt")
    stage1_classifier.save(output / "classifier_stage1.pt")
    stage2_classifier.save(output / "classifier_stage2.pt")
    torch.save({"stage1": stage1_results, "stage2": stage2_results}, output / "classifier_search.pt")
    classifier_search_stages = {
        "stage1": {"best_params": stage1_best["params"], "validation_metrics": {
            "accuracy": stage1_best["validation_accuracy"], "nll": stage1_best["validation_nll"],
            "brier": stage1_best["validation_brier"]}, "structural_test_metrics": stage1_test,
            "model_metadata": {"prototype_count": stage1_best["num_prototypes"]}},
        "stage2": {"best_params": stage2_best["params"], "validation_metrics": {
            "accuracy": stage2_best["validation_accuracy"], "nll": stage2_best["validation_nll"],
            "brier": stage2_best["validation_brier"]}, "structural_test_metrics": stage2_test,
            "model_metadata": {"prototype_count": stage2_best["num_prototypes"]}},
        "selected_stage": selected_stage, "selected_params": selected_record["params"],
        "selected_validation_score": selected_record["validation_accuracy"],
        "selected_structural_test_metrics": instantaneous,
    }
    results = {
        "schema_version": 1, "method": "RBF Prototype", "require_feature": True,
        "best_hyperparameters": selected_record["params"],
        "validation_score": selected_record["validation_accuracy"],
        "instantaneous": instantaneous,
        "classifier_search_stages": classifier_search_stages,
        "model": {"prototype_count": selected_record["num_prototypes"],
                  "model_size_bytes": (output / "classifier.pt").stat().st_size},
        "runtime_seconds": {"search_or_training": search_runtime, "bayes_search": bayes_runtime},
        "bayes": legacy_bayes, "sequential_search": sequential_search, "data_files": files,
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
