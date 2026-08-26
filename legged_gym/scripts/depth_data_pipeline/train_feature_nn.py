from __future__ import annotations

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    NeuralClassifierAdapter,
)
from .util_func import (
    classifier_metrics_from_scores, collect_engineered_scores,
    extract_dataset_features, fit_nn, fit_standardizer,
    json_safe, load_training_files, make_terrain_extractor, save_results, sequence_ids_for,
    processing_batch_size, run_staged_sequential_pipeline,
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
        self.activation_fn = activation_fn

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

        Note: `activation_fn` is returned as the live module instance actually used
        (the same object stored on `self`), not a fresh copy, since activation modules
        like `nn.ReLU()` are stateless and safe to reuse. If you need a JSON-serializable
        summary instead (e.g. for metadata.json), use `activation_fn.__class__.__name__`.
        """
        return {
            "cls": self.__class__.__name__,
            "feature_input_dim": self.feature_input_dim,
            "mlp_layer_dims": list(self.mlp_layer_dims),
            "activation_fn": self.activation_fn,
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
    classifier.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bayes_start = time.perf_counter()
    ordered = torch.load(files["ordered_test"], map_location="cpu", weights_only=False)
    ordered_scores = collect_engineered_scores(
        classifier, extractor, standardizer, ordered,
        chunk_size=processing_batch_size(args, len(ordered["labels"])),
    )
    sequential_search, filter_results, legacy_bayes = run_staged_sequential_pipeline(
        classifier, calibration_scores, calibration_labels,
        filter_validation_scores, filter_validation_labels, filter_validation_ids,
        ordered_scores, ordered["labels"], sequence_ids_for(ordered), output,
    )
    bayes_runtime = time.perf_counter() - bayes_start
    selected_temperature = legacy_bayes["selected_temperature"]
    classifier.temperature = selected_temperature
    ordered_instantaneous, bayesian = legacy_bayes["unfiltered_metrics"], legacy_bayes["metrics"]
    extractor.save(output / "extractor.pt")
    standardizer.save(output / "standardizer.pt")
    classifier.save(output / "classifier.pt")
    torch.save(nn_feature_model.get_args(), output / "nn_model_args.pt")
    best_validation = max(classifier.training_history.get("validation_accuracy", [float("nan")]))
    classifier_search_stages = {
        "base": {"best_params": {"epochs": 20, "batch_size": train_batch, "optimizer": "adam", "lr": 1e-3},
                 "validation_metrics": {"accuracy": best_validation},
                 "structural_test_metrics": instantaneous,
                 "model_metadata": {"parameter_count": sum(p.numel() for p in nn_feature_model.parameters())}},
        "selected_stage": "base", "selected_params": {"epochs": 20, "batch_size": train_batch,
        "optimizer": "adam", "lr": 1e-3}, "selected_validation_score": best_validation,
        "selected_structural_test_metrics": instantaneous,
    }
    results = {
        "schema_version": 1, "method": "feature NN", "require_feature": True,
        "best_hyperparameters": {"epochs": 20, "batch_size": train_batch, "optimizer": "adam", "lr": 1e-3},
        "validation_score": best_validation, "instantaneous": instantaneous,
        "classifier_search_stages": classifier_search_stages,
        "model": {"parameter_count": sum(p.numel() for p in nn_feature_model.parameters()),
                  "model_size_bytes": (output / "classifier.pt").stat().st_size},
        "runtime_seconds": {"search_or_training": training_runtime, "bayes_search": bayes_runtime},
        "training_history": classifier.training_history,
        "bayes": legacy_bayes, "sequential_search": sequential_search,
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
    
