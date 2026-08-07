from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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
        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, depth_image):
        return self.cnn(depth_image)

def fit_nn(
    self: NeuralClassifierAdapter, 
    inputs, 
    labels,
    val=None,
    epochs=20,
    lr=1e-3,
):
    device = self.device
    model = self.model
    
    if isinstance(labels, list):
        assert len(labels) == inputs.shape[0]
        self.set_class_ids(list(set(labels)))
        labels = torch.tensor(
            [self.class_to_index[c] for c in labels],
            dtype=torch.long
        )
    assert labels.shape[0] == inputs.shape[0]
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
            assert len(val_labels) == val_input.shape[0]
            val_labels = torch.tensor(
                [self.class_to_index[c] for c in val_labels],
                dtype=torch.long
            )
        assert val_labels.shape[0] == val_input.shape[0]
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

    train = torch.load(train_file)
    val = torch.load(validation_file)

    train_images = train["depth_images"].unsqueeze(1).float()
    train_labels = train["labels"]
    val_images = val["depth_images"].unsqueeze(1).float()
    val_labels = val["labels"]

    nn_raw_depth_model = TerrainDepthClassifierNN(
        depth_image_resolution=train_images.shape[-2:],   # (H, W)
        cnn_input_channel=1,                              # grayscale
        cnn_channel_dims=[8, 16],
        cnn_strides=[1, 1],
        cnn_fc_layer_dims=[128, len(set(train_labels))],
        cnn_kernel_sizes=[5, 3],
        cnn_activation_fn=nn.ELU(),
    )

    nn_raw_depth_classifier = NeuralClassifierAdapter(
        model = nn_raw_depth_model,
        class_ids=[],
        input_transform=None,
        fit_callback=fit_nn,
    )

    nn_raw_depth_classifier.fit(inputs=train_images, labels=train_labels, val=(val_images, val_labels), epochs=1)

    test = torch.load(test_file)
    test_images = test["depth_images"].unsqueeze(1).float()
    test_labels = test["labels"]

    metrics = nn_raw_depth_classifier.evaluate(test_images, test_labels)
    metrics.pop("labels")
    metrics.pop("predictions")
    print(metrics)

if __name__ == "__main__":
    main()