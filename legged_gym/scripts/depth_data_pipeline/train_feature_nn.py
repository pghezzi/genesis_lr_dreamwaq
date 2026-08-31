from __future__ import annotations

from .util_func import (
    collect_engineered_logits, extract_dataset_features, fit_standardizer,
    get_activation_fn, json_safe, load_training_files, make_terrain_extractor, processing_batch_size,
    save_results, sequence_ids_for, train_uncertainty_aware_nn_suite,
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
        dropout_p=0.0,
    ):
        super().__init__()
        self.feature_input_dim = feature_input_dim
        self.mlp_layer_dims = list(mlp_layer_dims)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout_p = float(dropout_p)

        mlp_layers = []

        for i, dim in enumerate(mlp_layer_dims):
            in_dim = feature_input_dim if i == 0 else mlp_layer_dims[i - 1]

            mlp_layers.append(nn.Linear(in_dim, dim))

            # Don't add activation after final layer
            if i != len(mlp_layer_dims) - 1:
                mlp_layers.append(self.activation_fn)
                if self.dropout_p > 0:
                    mlp_layers.append(nn.Dropout(self.dropout_p))

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
            "dropout_p": self.dropout_p,
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

    split_data = {
        "structural_validation": validation,
        "structural_test": torch.load(files["structural_test"], map_location="cpu", weights_only=False),
        "ordered_validation": torch.load(files["bayes_validation"], map_location="cpu", weights_only=False),
        "ordered_test": torch.load(files["ordered_test"], map_location="cpu", weights_only=False),
    }
    def model_factory(dropout_p):
        return TerrainDepthFeatureClassifierNN(
            train_features.shape[1], [512, 256, 128, len(class_ids)], nn.ELU(), dropout_p)
    def collect(classifier, data, samples, mc):
        return collect_engineered_logits(
            classifier, extractor, standardizer, data,
            chunk_size=processing_batch_size(args, len(data["labels"])),
            mc_samples=samples, mc_dropout=mc)
    results = train_uncertainty_aware_nn_suite(
        architecture="feature_nn", model_factory=model_factory, class_ids=class_ids,
        train_inputs=train_features, train_labels=train["labels"],
        validation_inputs=validation_features, validation_labels=validation["labels"],
        split_data=split_data, collect_logits=collect,
        validation_sequence_ids=sequence_ids_for(split_data["ordered_validation"]),
        test_sequence_ids=sequence_ids_for(split_data["ordered_test"]), output=output,
        device=device, train_batch=train_batch, validation_batch=validation_batch)
    extractor.save(output / "extractor.pt")
    standardizer.save(output / "standardizer.pt")
    results.update(method="feature NN", require_feature=True, data_files=files,
                   extractor_path=str(output / "extractor.pt"),
                   standardizer_path=str(output / "standardizer.pt"))
    save_results(output / "results.json", results)
    torch.save(json_safe(results), output / "results.pt")
    print(f"Saved feature NN run to {output}")


if __name__ == "__main__":
    main()
    
