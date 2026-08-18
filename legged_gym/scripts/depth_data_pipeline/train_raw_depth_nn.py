from __future__ import annotations

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    NeuralClassifierAdapter, build_filter_from_search_result, search_bayes_filter_hyperparameters,
)
from .util_func import (
    classifier_metrics_from_scores, collect_raw_depth_scores, evaluate_bayes_from_scores,
    fit_nn, json_safe, load_training_files,
    save_results, sequence_ids_for,
    transition_training_kwargs, processing_batch_size,
)

import torch
import torch.nn as nn
import time
from datetime import datetime
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

class TerrainDepthClassifierNN(nn.Module):
    def __init__(
        self,
        depth_image_resolution,
        cnn_input_channel,
        cnn_channel_dims,
        cnn_strides,
        cnn_fc_layer_dims,
        cnn_kernel_sizes,
        cnn_activation_fn,
    ):
        super().__init__()


        self.depth_image_resolution = tuple(depth_image_resolution)
        self.cnn_input_channel = cnn_input_channel
        self.cnn_channel_dims = list(cnn_channel_dims)
        self.cnn_strides = list(cnn_strides)
        self.cnn_fc_layer_dims = list(cnn_fc_layer_dims)
        self.cnn_kernel_sizes = list(cnn_kernel_sizes)
        self.cnn_activation_fn = cnn_activation_fn

        in_channels = cnn_input_channel
        in_height, in_width = depth_image_resolution
        cnn_layers = []
        for i, out_channels in enumerate(cnn_channel_dims):
            cnn_layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=cnn_kernel_sizes[i],
                    stride=cnn_strides[i],
                )
            )
            if i != 0:
                cnn_layers.append(cnn_activation_fn)

            in_channels = out_channels

            # Output size after conv
            in_height = (in_height - cnn_kernel_sizes[i]) // cnn_strides[i] + 1
            in_width = (in_width - cnn_kernel_sizes[i]) // cnn_strides[i] + 1

            # MaxPool after first conv
            if i == 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                cnn_layers.append(cnn_activation_fn)

                in_height = (in_height - 2) // 2 + 1
                in_width = (in_width - 2) // 2 + 1

        cnn_layers.append(nn.Flatten())

        cnn_out_dim = in_height * in_width * cnn_channel_dims[-1]

        for l, dim in enumerate(cnn_fc_layer_dims):
            in_dim = cnn_out_dim if l == 0 else cnn_fc_layer_dims[l - 1]
            cnn_layers.append(nn.Linear(in_dim, dim))
            cnn_layers.append(cnn_activation_fn)

        self._output_size = cnn_fc_layer_dims[-1]
        self.model = nn.Sequential(*cnn_layers)

    def forward(self, depth_image):
        return self.model(depth_image)
    
    def get_args(self) -> dict:
        """Return constructor kwargs sufficient to rebuild an equivalent (untrained)
        instance via `TerrainDepthClassifierNN(**model.get_args())`.

        Note: `cnn_activation_fn` is returned as the live module instance actually used
        (reused across all conv/fc layers already, so this matches original construction).
        For a JSON-serializable summary instead, use `cnn_activation_fn.__class__.__name__`.
        """
        return {
            "cls": self.__class__.__name__,
            "depth_image_resolution": tuple(self.depth_image_resolution),
            "cnn_input_channel": self.cnn_input_channel,
            "cnn_channel_dims": list(self.cnn_channel_dims),
            "cnn_strides": list(self.cnn_strides),
            "cnn_fc_layer_dims": list(self.cnn_fc_layer_dims),
            "cnn_kernel_sizes": list(self.cnn_kernel_sizes),
            "cnn_activation_fn": self.cnn_activation_fn,
        }

def main():
    args, files = load_training_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output_dir or (Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
                                 f"raw_depth_nn_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    train = torch.load(files["train"], map_location="cpu", weights_only=False)
    structural_training_images = train["depth_images"].unsqueeze(1).float()
    structural_training_labels = train["labels"]

    validation = torch.load(files["validation"], map_location="cpu", weights_only=False)
    structural_validation_images = validation["depth_images"].unsqueeze(1).float()
    structural_validation_labels = validation["labels"]
    train_batch = processing_batch_size(args, len(structural_training_labels))
    validation_batch = processing_batch_size(args, len(structural_validation_labels))

    label_values = structural_training_labels.tolist() if torch.is_tensor(structural_training_labels) else list(structural_training_labels)
    class_ids = list(dict.fromkeys(label_values))
    nn_raw_depth_model = TerrainDepthClassifierNN(
        depth_image_resolution=structural_training_images.shape[-2:],   # (H, W)
        cnn_input_channel=1,                              # grayscale
        cnn_channel_dims=[8, 16],
        cnn_strides=[1, 1],
        cnn_fc_layer_dims=[128, len(class_ids)],
        cnn_kernel_sizes=[5, 3],
        cnn_activation_fn=nn.ELU(),
    )

    classifier = NeuralClassifierAdapter(
        model = nn_raw_depth_model,
        class_ids=class_ids,
        input_transform=None,
        fit_callback=fit_nn,
        device=device,
    )

    training_start = time.perf_counter()
    classifier.fit(inputs=structural_training_images, labels=structural_training_labels,
                   val=(structural_validation_images, structural_validation_labels), epochs=20,
                   batch_size=train_batch, validation_batch_size=validation_batch)
    training_runtime = time.perf_counter() - training_start


    del train, structural_training_images, structural_training_labels
    del validation, structural_validation_images, structural_validation_labels

    test = torch.load(files["structural_test"], map_location="cpu", weights_only=False)
    structural_test_labels = test["labels"]
    structural_test_scores = collect_raw_depth_scores(
        classifier, test["depth_images"],
        chunk_size=processing_batch_size(args, len(structural_test_labels)),
    )
    instantaneous = classifier_metrics_from_scores(
        classifier, structural_test_scores, structural_test_labels
    )
    del test, structural_test_scores, structural_test_labels

    manual_prior = {
        label: 1.0 / len(classifier.class_ids)
        for label in classifier.class_ids
    }

    calibration = torch.load(files["calibration"], map_location="cpu", weights_only=False)
    calibration_labels = classifier._normalize_labels(calibration["labels"])
    calibration_scores = collect_raw_depth_scores(
        classifier, calibration["depth_images"],
        chunk_size=processing_batch_size(args, len(calibration_labels)),
    )
    del calibration
    validation = torch.load(files["bayes_validation"], map_location="cpu", weights_only=False)
    filter_validation_labels = classifier._normalize_labels(validation["labels"])
    filter_validation_ids = sequence_ids_for(validation)
    filter_validation_scores = collect_raw_depth_scores(
        classifier, validation["depth_images"],
        chunk_size=processing_batch_size(args, len(filter_validation_labels)),
    )
    del validation

    transition_kwargs = transition_training_kwargs(files)
    classifier.to("cpu")
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
        device="cpu",
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
    ordered_scores = collect_raw_depth_scores(
        classifier, ordered["depth_images"],
        chunk_size=processing_batch_size(args, len(ordered["labels"])),
    )
    ordered_instantaneous = classifier_metrics_from_scores(
        classifier, ordered_scores, ordered["labels"], selected_temperature
    )
    bayesian = evaluate_bayes_from_scores(
        classifier, bayes_filter, ordered_scores, ordered["labels"],
        sequence_ids_for(ordered), selected_temperature,
    )
    bayes_filter.save(output / "bayes_filter.pt")
    classifier.save(output / "classifier.pt")
    torch.save(nn_raw_depth_model.get_args(), output / "nn_model_args.pt")
    torch.save(filter_results, output / "bayes_search.pt")
    best_validation = max(classifier.training_history.get("validation_accuracy", [float("nan")]))
    results = {
        "schema_version": 1, "method": "raw-depth NN", "require_feature": False,
        "best_hyperparameters": {"epochs": 20, "batch_size": train_batch, "optimizer": "adam", "lr": 1e-3},
        "validation_score": best_validation, "instantaneous": instantaneous,
        "model": {"parameter_count": sum(p.numel() for p in nn_raw_depth_model.parameters()),
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
    print(f"Saved raw-depth NN run to {output}")

if __name__ == "__main__":
    main()
