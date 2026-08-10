from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter, PCAWhitenedRBFSVM
from util_func import fit_nn, evaluate_classifier, extract_in_chunks, make_terrain_extractor

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

        mlp_layers = []

        for i, dim in enumerate(mlp_layer_dims):
            in_dim = feature_input_dim if i == 0 else mlp_layer_dims[i - 1]

            mlp_layers.append(nn.Linear(in_dim, dim))

            # Don't add activation after final layer
            if i != len(mlp_layer_dims) - 1:
                mlp_layers.append(activation_fn)

        self._output_size = mlp_layer_dims[-1]
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, features):
        return self.mlp(features)

def train_feature_nn_from_data_set(train_file, test_file, validation_file, extractor, *_, **__):
    train = torch.load(train_file)
    val = torch.load(validation_file)

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
    train_feature_nn_from_data_set(train_file, test_file, validation_file, extractor)


