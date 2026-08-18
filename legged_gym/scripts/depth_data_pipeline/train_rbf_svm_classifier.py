"""Train a standardized-feature RBF SVM and its Bayesian sequence filter."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    RBFSVM, build_filter_from_search_result, search_bayes_filter_hyperparameters,
    search_rbf_svm_hyperparameters,
)
from .util_func import (
    classifier_metrics_from_scores, collect_engineered_scores, evaluate_bayes_from_scores,
    extract_dataset_features, fit_standardizer,
    json_safe, load_training_files, make_terrain_extractor, save_results, sequence_ids_for,
    transition_training_kwargs, processing_batch_size,
)


def main() -> None:
    args, files = load_training_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output_dir or (
        Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
        f"rbf_svm_{datetime.now():%Y%m%d_%H%M%S}"
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
    classifier, search_results = search_rbf_svm_hyperparameters(
        train_features, train["labels"], validation_features, validation["labels"],
        base_config={
            "feature_dim": train_features.shape[1], "device": device,
            "dtype": torch.float32, "epochs": 300, "batch_size": train_batch,
            "early_stopping_patience": 30, "random_seed": 0,
        },
        search_space={
            "gamma": ["scale", 0.01, 0.05, 0.1],
            "max_kernel_samples": [256, 512],
            "weight_decay": [1e-4, 1e-3, 1e-2],
            "learning_rate": [5e-3, 1e-2],
            "squared_hinge": [True], "class_balance": [True],
        },
    )
    search_runtime = time.perf_counter() - search_start

    structural_test = torch.load(files["structural_test"], map_location="cpu", weights_only=False)
    structural_test_scores = collect_engineered_scores(
        classifier, extractor, standardizer, structural_test,
        chunk_size=processing_batch_size(args, len(structural_test["labels"])),
    )
    instantaneous = classifier_metrics_from_scores(
        classifier, structural_test_scores, structural_test["labels"]
    )
    del train, validation, train_features, validation_features
    del structural_test, structural_test_scores

    calibration = torch.load(files["calibration"], map_location="cpu", weights_only=False)
    calibration_labels = classifier._normalize_labels(calibration["labels"])
    calibration_scores = collect_engineered_scores(
        classifier, extractor, standardizer, calibration,
        chunk_size=processing_batch_size(args, len(calibration_labels)),
    )
    del calibration
    bayes_validation = torch.load(files["bayes_validation"], map_location="cpu", weights_only=False)
    bayes_validation_ids = sequence_ids_for(bayes_validation)
    bayes_validation_labels = classifier._normalize_labels(bayes_validation["labels"])
    bayes_validation_scores = collect_engineered_scores(
        classifier, extractor, standardizer, bayes_validation,
        chunk_size=processing_batch_size(args, len(bayes_validation_labels)),
    )
    del bayes_validation
    prior = {label: 1.0 / len(classifier.class_ids) for label in classifier.class_ids}
    transition_kwargs = transition_training_kwargs(files)
    classifier.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bayes_start = time.perf_counter()
    filter_results = search_bayes_filter_hyperparameters(
        classifier, None, bayes_validation_labels, prior,
        sequence_ids=bayes_validation_ids,
        filter_scores=bayes_validation_scores,
        observation_calibration_scores=calibration_scores,
        observation_calibration_labels=calibration_labels,
        temperatures=[0.5, 0.75, 1.0, 1.5, 2.0],
        stay_probabilities=[0.90, 0.94, 0.97], evidence_powers=[0.50, 0.75, 1.0],
        transition_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        min_evidence_powers=[0.10, 0.25], confidence_gammas=[1.0, 2.0],
        observation_modes=["soft"], observation_pseudocounts=[0.5],
        scoring="balanced_accuracy", device="cpu",
        **transition_kwargs,
    )
    bayes_runtime = time.perf_counter() - bayes_start
    del bayes_validation_scores, calibration_scores, transition_kwargs
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
        "schema_version": 1, "method": "RBF SVM", "require_feature": True,
        "best_hyperparameters": search_results[0].params,
        "validation_score": search_results[0].validation_accuracy,
        "instantaneous": instantaneous,
        "model": {"kernel_basis_size": int(classifier.kernel_basis.shape[0]),
                  "model_size_bytes": (output / "classifier.pt").stat().st_size},
        "runtime_seconds": {"search_or_training": search_runtime, "bayes_search": bayes_runtime},
        "bayes": {
            "selected_temperature": temperature,
            "filter_parameters": {k: selected[k] for k in (
                "stay_probability", "evidence_power", "min_evidence_power",
                "transition_alpha", "transition_matrix", "transition_source",
                "confidence_gamma", "observation_mode", "observation_pseudocount"
            )},
            "validation_score": selected["objective"],
            "unfiltered_metrics": ordered_instantaneous, "metrics": bayesian,
        },
        "data_files": files,
    }
    save_results(output / "results.json", results)
    save_results(output / "params.json", {
        "classifier": results["best_hyperparameters"],
        "temperature": temperature, "bayes_filter": results["bayes"]["filter_parameters"],
    })
    save_results(output / "metrics.json", {
        "instantaneous": instantaneous, "ordered_instantaneous": ordered_instantaneous,
        "bayesian": bayesian,
        "validation_score": results["validation_score"],
        "bayes_validation_score": results["bayes"]["validation_score"],
    })
    torch.save(json_safe(results), output / "results.pt")
    print(f"Saved RBF SVM run to {output}")
    print(f"Structural test accuracy: {instantaneous['accuracy']:.4f}")
    print(f"Bayesian ordered-test accuracy: {bayesian['accuracy']:.4f}")


if __name__ == "__main__":
    main()
