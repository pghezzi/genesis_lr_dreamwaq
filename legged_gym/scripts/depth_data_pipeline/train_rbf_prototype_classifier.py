from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
    PCAWhitenedRBFPrototypeClassifier, search_prototype_rbf_hyperparameters_dataloader, search_bayes_filter_hyperparameters
)
from util_func import evaluate_classifier, extract_in_chunks, make_terrain_extractor, save_classifier

from collections.abc import Sequence


import torch
from torch.utils.data import Dataset, DataLoader

class TerrainFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: Sequence):
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError("features must have shape [N, feature_dim]")
        if features.shape[0] != len(labels):
            raise ValueError("feature and label counts differ")

        self.features = features
        self.labels = list(labels)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def make_loader(
    features: torch.Tensor,
    labels: Sequence,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TerrainFeatureDataset(features, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


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

    validation = torch.load(structural_training_data)
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

    def make_train_loader():
        return make_loader(
            structural_training_features,
            structural_training_labels,
            batch_size=256,
            shuffle=True,
        )


    def make_validation_loader():
        return make_loader(
            structural_validation_features,
            structural_validation_labels,
            batch_size=512,
            shuffle=False,
        )

    best_classifier, prototype_search_results = (
        search_prototype_rbf_hyperparameters_dataloader(
            train_loader_factory=make_train_loader,
            validation_loader_factory=make_validation_loader,
            base_config={
                "feature_dim": train_features.shape[1],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "dtype": torch.float32,
                "kmeans_fit_mode": "mini_batch",
                "store_training_data": False,
                "random_seed": 0,
            },
            search_space={
                "pca_dim": [16, 24, 32],
                "prototypes_per_class": [4, 8, 12],
                "gamma": ["scale", 0.25, 0.5, 1.0],
                "metric_type": ["euclidean", "diag_mahalanobis"],
                "aggregation": ["logsumexp", "max"],
                "prototype_init": ["kmeans++", "farthest"],
                "prototype_epochs": [5, 10],
                "initialization_sample_size": [1024, 2048],
                "reset_counts_each_epoch": [True],
                "variance_shrinkage": [0.1, 0.3],
            },
            scoring="validation_accuracy",
            verbose=False,
        )
    )

    best_trial = prototype_search_results[0]
    print("Best structural parameters:", best_trial.params)
    print("Validation accuracy:", best_trial.validation_accuracy)
    print("Validation NLL:", best_trial.validation_nll)
    print("Validation Brier score:", best_trial.validation_brier)
    print("Total prototypes:", best_trial.num_prototypes)

    classifier = best_classifier
    
    del train, structural_training_features, structural_training_labels
    del validation, structural_validation_features, structural_validation_labels

    test = torch.load(structural_test_data)
    structural_test_features = extract_in_chunks(
        extractor,
        test["depth_images"],
        test["orientation_rpy"],
        test["angular_velocity"],
        chunk_size=test["depth_images"].shape[0]
    )
    structural_test_labels = test["labels"]

    acc = evaluate_classifier(classifier, structural_test_features, structural_test_labels)

    del test, structural_test_features, structural_test_labels

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
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "depth_waq_selector", "full_models", f"rbf_model_{timestamp}")
    extractor.save(os.path.join(out_dir, "extractor.pt"))
    bayes_filter.save(os.path.join(out_dir, "bayes_filter.pt"))
    classifier.save(os.path.join(out_dir, "classifier.pt"))
    

if __name__ == "__main__":
    main()
