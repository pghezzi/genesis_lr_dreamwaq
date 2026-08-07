
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter, PCAWhitenedRBFSVM

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


# TRAINER


def main():
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

    train = torch.load(train_file)
    val = torch.load(validation_file)

    train_images = train["depth_images"].unsqueeze(1).float()
    train_labels = train["labels"]
    val_images = val["depth_images"].unsqueeze(1).float()
    val_labels = val["labels"]

    extractor = PCAWhitenedRBFPrototypeClassifier(feature_dim=RAW_DIM, pca_dim=32)

    nn_feature_model = TerrainDepthClassifierNN(
        feature_input_dim=None,
        mlp_layer_dims=[512, 256, 128, len(set(train_labels))],
        activation_fn=nn.ELU(),
    )