from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import PCAWhitenedRBFPrototypeClassifier, search_prototype_rbf_hyperparameters_dataloader
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




def train_rbf_prototype_from_data_set(train_file, test_file, validation_file, extractor, *_, **__):
    validation = torch.load(validation_file)
    validation_features = extract_in_chunks(
        extractor,
        validation["depth_images"],
        validation["orientation_rpy"],
        validation["angular_velocity"],
        chunk_size=validation["depth_images"].shape[0]
    )
    validation_labels = validation["labels"]

    train = torch.load(train_file)
    train_features = extract_in_chunks(
        extractor,
        train["depth_images"],
        train["orientation_rpy"],
        train["angular_velocity"],
        chunk_size=256,   # tune down if still OOMing
    )
    train_labels = train["labels"]

    print("Past Training")

    def make_train_loader():
        return make_loader(
            train_features,
            train_labels,
            batch_size=256,
            shuffle=True,
        )


    def make_validation_loader():
        return make_loader(
            validation_features,
            validation_labels,
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

    return classifier

    #out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models"
    #os.makedirs(out_dir, exist_ok=True)
    #
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #model_path = os.path.join(out_dir, f"feature_classifier_acc_{str(acc).replace('.', '_')}_{timestamp}.pt")
    #
    #classifier.save(model_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create model using data")
    parser.add_argument("--folder", type=str, default=None, help="folder with data")
    args = parser.parse_args() 
    from pathlib import Path 
    folder = Path(args.folder)
    files = { name : path for name in ("calibration", "val", "train", "test") if (path := folder / f"{name}.pt").is_file() } 
    
    train_file = files["train"]
    test_file = files["test"]
    validation_file = files["val"]
    calibration_file = files["calibration"]
    extractor = make_terrain_extractor(calibration_file)
    classifier = train_rbf_prototype_from_data_set(train_file, test_file, validation_file, extractor)

    save_classifier(classifier, files, extractor)

    print(f"Saved classifier to: {classifier_dir}")


