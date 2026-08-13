from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter, PCAWhitenedRBFSVM
from util_func import fit_nn, evaluate_classifier, extract_in_chunks, make_terrain_extractor, save_classifier

import torch
import torch.nn as nn

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
            "cls": self.__name__,
            "feature_input_dim": self.feature_input_dim,
            "mlp_layer_dims": list(self.mlp_layer_dims),
            "activation_fn": self.activation_fn,
        }

def main():
    from .util_func import get_files_for_training

    (
        structural_training_data,
        structural_validation_data,
        observation_calibration_data,
        structural_test_data,
        ordered_filter_validation,
        final_test_sequences,
    ) = get_files_for_training()

    extractor = make_terrain_extractor(observation_calibration_data)

    validation = torch.load(structural_validation_data)
    structural_validation_features = extract_in_chunks(
        extractor,
        validation["depth_images"],
        validation["orientation_rpy"],
        validation["angular_velocity"],
        chunk_size=validation["depth_images"].shape[0]
    )
    structural_validation_labels = validation["labels"]

    train = torch.load(structural_training_data)
    structural_training_features = extract_in_chunks(
        extractor,
        train["depth_images"],
        train["orientation_rpy"],
        train["angular_velocity"],
        chunk_size=256,   # tune down if still OOMing
    )
    structural_training_labels = train["labels"]

    nn_feature_model = TerrainDepthFeatureClassifierNN(
        feature_input_dim=extractor.feature_dim,
        mlp_layer_dims=[512, 256, 128, len(set(train_labels))],
        activation_fn=nn.ELU(),
    )

    classifier = NeuralClassifierAdapter(
        model = nn_feature_model,
        class_ids=[],
        input_transform=None,
        fit_callback=fit_nn,
    )

    classifier.require_feature = True

    classifier.fit(inputs=train_features, labels=train_labels, val=(validation_features, validation_labels), epochs=20)

    test = torch.load(test_file)
    test_features = extract_in_chunks(
        extractor,
        test["depth_images"],
        test["orientation_rpy"],
        test["angular_velocity"],
        chunk_size=test["depth_images"].shape[0]
    )
    test_labels = test["labels"]

    acc = evaluate_classifier(classifier, test_features, test_labels)

    manual_prior = {
        label: 1.0 / len(classifier.class_ids)
        for label in classifier.class_ids
    }

    validation = torch.load(ordered_filter_validation)
    filter_validation_features = extract_in_chunks(
        extractor,
        validation["depth_images"],
        validation["orientation_rpy"],
        validation["angular_velocity"],
        chunk_size=256,   # tune down if still OOMing
    )
    filter_validation_labels = validation["labels"]

    filter_validation_episode_ids = torch.arange(filter_validation_features.shape[0] // validation["per_eps"]).repeat_interleave(validation["per_eps"])

    filter_results = search_bayes_filter_hyperparameters(
        classifier=classifier,
        filter_inputs=filter_validation_features,
        true_labels=filter_validation_labels,
        manual_prior=manual_prior,
        sequence_ids=filter_validation_episode_ids,
        temperatures=[0.5, 0.75, 1.0, 1.5, 2.0],
        stay_probabilities=[0.90, 0.94, 0.97],
        evidence_powers=[0.50, 0.75, 1.0],
        min_evidence_powers=[0.10, 0.25],
        confidence_gammas=[1.0, 2.0],
        observation_modes=["soft"],
        observation_pseudocounts=[0.5],
        scoring="balanced_accuracy",
    )

    del validation, filter_validation_features, filter_validation_labels

    best_filter = filter_results[0]
    bayes_filter, selected_temperature = build_filter_from_search_result(
        classifier,
        best_filter,
        manual_prior,
    )

    test_probabilities, ordered_labels = classifier.predict_class_distribution(
        test_sequence_features,
        temperature=selected_temperature,
        probability_floor=1e-8,
    )

    test_predictions, test_posteriors, test_evidence = run_filter_sequences(
        bayes_filter,
        test_probabilities,
        sequence_ids=test_sequence_episode_ids,
    )

    test_metrics = evaluate_predictions(
        test_sequence_labels,
        test_predictions,
        ordered_labels,
        sequence_ids=test_sequence_episode_ids,
    )

    print(test_metrics.as_dict())

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "depth_waq_selector", "full_models", f"feature_nn_model_{timestamp}")
    extractor.save(os.path.join(out_dir, "extractor.pt"))
    bayes_filter.save(os.path.join(out_dir, "bayes_filter.pt"))
    classifier.save(os.path.join(out_dir, "classifier.pt"))
    torch.save(nn_feature_model.get_args(), os.path.join(out_dir, "nn_model_args.pt"))


if __name__ == "__main__":
    main()
    


