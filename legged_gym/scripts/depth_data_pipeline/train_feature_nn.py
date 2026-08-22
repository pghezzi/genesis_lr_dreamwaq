from __future__ import annotations

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    NeuralClassifierAdapter, build_filter_from_search_result, search_bayes_filter_hyperparameters,
)
from .util_func import (
    classifier_metrics_from_scores, collect_engineered_scores, evaluate_bayes_from_scores,
    extract_dataset_features, fit_nn, fit_standardizer,
    json_safe, load_training_files, make_terrain_extractor, save_results, sequence_ids_for,
    transition_training_kwargs, processing_batch_size,
    get_activation_fn
)

import torch
import torch.nn as nn
import time
from datetime import datetime
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

class TerrainDepthFeatureClassifierNN(nn.Module):
    def __init__(
        self,
        feature_input_dim,
        mlp_layer_dims,
        activation_fn,
    ):
        super().__init__()
        self.feature_input_dim = feature_input_dim
        self.mlp_layer_dims = list(mlp_layer_dims)
        self.activation_fn = get_activation_fn(activation_fn)

        mlp_layers = []

        for i, dim in enumerate(mlp_layer_dims):
            in_dim = feature_input_dim if i == 0 else mlp_layer_dims[i - 1]

            mlp_layers.append(nn.Linear(in_dim, dim))

            # Don't add activation after final layer
            if i != len(mlp_layer_dims) - 1:
                mlp_layers.append(activation_fn)

        self._output_size = mlp_layer_dims[-1]
        self.model = nn.Sequential(*mlp_layers)

    def forward(self, features):
        return self.model(features)
    
    def get_args(self) -> dict:
        """Return constructor kwargs sufficient to rebuild an equivalent (untrained)
        instance via `TerrainDepthFeatureClassifierNN(**model.get_args())`.
        """
        return {
            "cls": self.__class__.__name__,
            "feature_input_dim": self.feature_input_dim,
            "mlp_layer_dims": list(self.mlp_layer_dims),
            "activation_fn": self.activation_fn.__class__.__name__,
        }

def main():
    args, files = load_training_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output_dir or (Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
                                 f"feature_nn_{datetime.now():%Y%m%d_%H%M%S}")
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
    label_values = train["labels"].tolist() if torch.is_tensor(train["labels"]) else list(train["labels"])
    class_ids = list(dict.fromkeys(label_values))

    nn_feature_model = TerrainDepthFeatureClassifierNN(
        feature_input_dim=train_features.shape[1],
        mlp_layer_dims=[512, 256, 128, len(class_ids)],
        activation_fn=nn.ELU(),
    )

    classifier = NeuralClassifierAdapter(
        model = nn_feature_model,
        class_ids=class_ids,
        input_transform=None,
        fit_callback=fit_nn,
        device=device,
    )

    classifier.require_feature = True

    training_start = time.perf_counter()
    classifier.fit(inputs=train_features, labels=train["labels"],
                   val=(validation_features, validation["labels"]), epochs=20,
                   batch_size=train_batch, validation_batch_size=validation_batch)
    training_runtime = time.perf_counter() - training_start
    test = torch.load(files["structural_test"], map_location="cpu", weights_only=False)
    test_scores = collect_engineered_scores(
        classifier, extractor, standardizer, test,
        chunk_size=processing_batch_size(args, len(test["labels"])),
    )
    instantaneous = classifier_metrics_from_scores(classifier, test_scores, test["labels"])
    del train, validation, train_features, validation_features, test, test_scores

    manual_prior = {
        label: 1.0 / len(classifier.class_ids)
        for label in classifier.class_ids
    }

    calibration = torch.load(files["calibration"], map_location="cpu", weights_only=False)
    calibration_labels = classifier._normalize_labels(calibration["labels"])
    calibration_scores = collect_engineered_scores(
        classifier, extractor, standardizer, calibration,
        chunk_size=processing_batch_size(args, len(calibration_labels)),
    )
    del calibration
    validation = torch.load(files["bayes_validation"], map_location="cpu", weights_only=False)
    filter_validation_labels = classifier._normalize_labels(validation["labels"])
    filter_validation_ids = sequence_ids_for(validation)
    filter_validation_scores = collect_engineered_scores(
        classifier, extractor, standardizer, validation,
        chunk_size=processing_batch_size(args, len(filter_validation_labels)),
    )
    del validation
    transition_kwargs = transition_training_kwargs(files)
    classifier.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bayes_start = time.perf_counter()
    filter_results = search_bayes_filter_hyperparameters(
        classifier=classifier,
        filter_inputs=None,
        filter_scores=filter_validation_scores,
        true_labels=filter_validation_labels,
        manual_prior=manual_prior,
        sequence_ids=filter_validation_ids,
        observation_calibration_scores=calibration_scores,
        observation_calibration_labels=calibration_labels,
        temperatures=[0.5, 0.75, 1.0, 1.5, 2.0],
        stay_probabilities=[0.90, 0.94, 0.97],
        transition_alphas=[0.0, 0.25, 0.5, 0.75, 1.0],
        evidence_powers=[0.50, 0.75, 1.0],
        min_evidence_powers=[0.10, 0.25],
        confidence_gammas=[1.0, 2.0],
        observation_modes=["soft"],
        observation_pseudocounts=[0.5],
        scoring="balanced_accuracy",
        device=device,
        **transition_kwargs,
    )
    bayes_runtime = time.perf_counter() - bayes_start
    del filter_validation_scores, calibration_scores, transition_kwargs

    best_filter = filter_results[0]
    bayes_filter, selected_temperature = build_filter_from_search_result(
        classifier,
        best_filter,
        manual_prior,
        device=device,
    )
    classifier.temperature = selected_temperature
    ordered = torch.load(files["ordered_test"], map_location="cpu", weights_only=False)
    ordered_scores = collect_engineered_scores(
        classifier, extractor, standardizer, ordered,
        chunk_size=processing_batch_size(args, len(ordered["labels"])),
    )
    ordered_instantaneous = classifier_metrics_from_scores(
        classifier, ordered_scores, ordered["labels"], selected_temperature
    )
    bayesian = evaluate_bayes_from_scores(
        classifier, bayes_filter, ordered_scores, ordered["labels"],
        sequence_ids_for(ordered), selected_temperature,
    )
    extractor.save(output / "extractor.pt")
    standardizer.save(output / "standardizer.pt")
    bayes_filter.save(output / "bayes_filter.pt")
    classifier.save(output / "classifier.pt")
    torch.save(nn_feature_model.get_args(), output / "nn_model_args.pt")
    torch.save(filter_results, output / "bayes_search.pt")
    best_validation = max(classifier.training_history.get("validation_accuracy", [float("nan")]))
    results = {
        "schema_version": 1, "method": "feature NN", "require_feature": True,
        "best_hyperparameters": {"epochs": 20, "batch_size": train_batch, "optimizer": "adam", "lr": 1e-3},
        "validation_score": best_validation, "instantaneous": instantaneous,
        "model": {"parameter_count": sum(p.numel() for p in nn_feature_model.parameters()),
                  "model_size_bytes": (output / "classifier.pt").stat().st_size},
        "runtime_seconds": {"search_or_training": training_runtime, "bayes_search": bayes_runtime},
        "training_history": classifier.training_history,
        "bayes": {"selected_temperature": selected_temperature,
                  "filter_parameters": {k: best_filter[k] for k in (
                      "stay_probability", "evidence_power", "min_evidence_power", "confidence_gamma",
                      "transition_alpha", "transition_matrix", "transition_source",
                      "observation_mode", "observation_pseudocount")},
                  "validation_score": best_filter["objective"],
                  "unfiltered_metrics": ordered_instantaneous, "metrics": bayesian},
        "data_files": files,
    }
    save_results(output / "results.json", results)
    save_results(output / "params.json", {
        "classifier": results["best_hyperparameters"], "temperature": selected_temperature,
        "bayes_filter": results["bayes"]["filter_parameters"],
    })
    save_results(output / "metrics.json", {"instantaneous": instantaneous,
                                            "ordered_instantaneous": ordered_instantaneous,
                                            "bayesian": bayesian})
    torch.save(json_safe(results), output / "results.pt")
    print(f"Saved feature NN run to {output}")


if __name__ == "__main__":
    main()
    
