
from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter, PCAWhitenedRBFSVM
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor

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

def fit_nn(
    self: NeuralClassifierAdapter, 
    inputs, 
    labels,
    val=None,
    extractor=None,
    epochs=20,
    lr=1e-3,
):
    device = self.device
    model = self.model
    if extractor:
        self.input_transform = lambda inputs :  extractor.extract_batch(*inputs)
        inputs = self.input_transform(inputs).clone()
    
    if isinstance(labels, list):
        self.set_class_ids(list(set(labels)))
        labels = self._encode_labels(labels)
    train_loader = DataLoader(
        TensorDataset(inputs, labels),
        batch_size=64,
        shuffle=True,
    )
    val_loader = None


    if val:
        val_input = val[0]
        val_labels = val[1]
        if isinstance(val_labels, list):
            val_labels = self._encode_labels(val_labels)
            val_input = self.input_transform(val_input) if self.input_transform else val_input
        val_loader = DataLoader(
            TensorDataset(val_input, val_labels),
            batch_size=256,
        )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        print(
            f"Epoch {epoch+1:3d} | "
            f"train loss {train_loss/train_total:.4f} | "
            f"train acc {train_correct/train_total:.4f}",
            end=""
        )

        if val_loader is not None:
            model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    logits = model(inputs)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            print(
                f" | val loss {val_loss/val_total:.4f}"
                f" | val acc {val_correct/val_total:.4f}"
            )
        else:
            print()


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
    calibration_file = files["calibration"]

    calibration = torch.load(calibration_file)
    calibration_images = calibration["depth_images"].unsqueeze(1).float()
    calibration_rpy = calibration["orientation_rpy"]

    extractor = SobelDepthTerrainFeatureExtractor(
        output_size=(calibration_images.shape[-2:]),
        min_depth=0.02,                  # was 0.10 m
        max_depth=1.0,                   # was 5.0 m
        far_depth=0.6,                   # was 3.0 m
        close_depth=0.15,                # was 0.75 m
        close_residual_threshold=0.05,   # was 0.25 m
        sobel_edge_threshold=0.007,      # was 0.035
        depth_scale=None,                # auto = max_depth - min_depth = 0.98
    )
    
    extractor.fit_reference_model(calibration_images, calibration_rpy)

    del calibration_images, calibration_rpy

    train = torch.load(train_file)
    val = torch.load(validation_file)

    train_images = train["depth_images"].unsqueeze(1).float()
    train_labels = train["labels"]
    train_rpy = train["orientation_rpy"]
    train_ang = train["angular_velocity"]

    n = train_images.size(0)
    half = n // 2

    train_images = train_images[:half]
    train_labels = train_labels[:half]
    train_rpy = train_rpy[:half]
    train_ang = train_ang[:half]

    val_images = val["depth_images"].unsqueeze(1).float()
    val_labels = val["labels"]
    val_rpy = val["orientation_rpy"]
    val_ang = val["angular_velocity"]

    nn_feature_model = TerrainDepthFeatureClassifierNN(
        feature_input_dim=extractor.feature_dim,
        mlp_layer_dims=[512, 256, 128, len(set(train_labels))],
        activation_fn=nn.ELU(),
    )

    nn_feature_classifier = NeuralClassifierAdapter(
        model = nn_feature_model,
        class_ids=[],
        input_transform=None,
        fit_callback=fit_nn,
    )

    nn_feature_classifier.fit(inputs=(train_images, train_rpy, train_ang), labels=train_labels, val=((val_images, val_rpy, val_ang), val_labels), extractor=extractor, epochs=1)

    test = torch.load(test_file)
    test_images = test["depth_images"].unsqueeze(1).float()
    test_rpy = val["orientation_rpy"]
    test_ang = val["angular_velocity"]
    test_labels = val["labels"]

    metrics = nn_feature_classifier.evaluate((test_images, test_rpy, test_ang), test_labels)

    metrics.pop("labels")
    metrics.pop("predictions")

    print(metrics)

if __name__ == "__main__":
    main()

