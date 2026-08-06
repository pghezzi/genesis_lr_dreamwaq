import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import List

class TerrainClassifierNN(nn.Module):
    """Encodes a depth image into a latent vector."""
    __constants__ = ["classes"]

    def __init__(
        self,
        depth_image_resolution,
        cnn_input_channel,
        cnn_channel_dims,
        cnn_strides,
        cnn_fc_layer_dims,
        cnn_kernel_sizes,
        cnn_activation_fn,
        classes: List[str],
    ):
        super().__init__()

        self.classes = classes
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

    @torch.jit.export
    def predict_depth(self, depth_image) -> str:
        logits = self.forward(depth_image)
        idx = int(torch.argmax(logits, dim=1)[0])
        return self.classes[idx]

    @torch.jit.export
    def class_name(self, idx: int) -> str:
        return self.classes[idx]

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def train_model(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for depth, labels in train_loader:
            depth = depth.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(depth)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * depth.size(0)
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
                for depth, labels in val_loader:
                    depth = depth.to(device)
                    labels = labels.to(device)

                    logits = model(depth)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * depth.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            print(
                f" | val loss {val_loss/val_total:.4f}"
                f" | val acc {val_correct/val_total:.4f}"
            )
        else:
            print()

    return model

def evaluate(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for depth, labels in loader:
            depth = depth.to(device)
            labels = labels.to(device)

            logits = model(depth)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test accuracy: {acc:.4f} ({correct}/{total})")
    return acc

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

    train_labels_str = train["labels"]
    val_labels_str = val["labels"]

    classes = sorted(set(train_labels_str) | set(val_labels_str))
    class_to_idx = {name: i for i, name in enumerate(classes)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    train_labels = torch.tensor(
        [class_to_idx[label] for label in train_labels_str],
        dtype=torch.long,
    )

    val_labels = torch.tensor(
        [class_to_idx[label] for label in val_labels_str],
        dtype=torch.long,
    )

    train_images = train["depth_images"].unsqueeze(1).float()
    val_images = val["depth_images"].unsqueeze(1).float()

    model = TerrainClassifierNN(
        depth_image_resolution=train_images.shape[-2:],   # (H, W)
        cnn_input_channel=1,                              # grayscale
        cnn_channel_dims=[8, 16],
        cnn_strides=[1, 1],
        cnn_fc_layer_dims=[128, len(torch.unique(train_labels))],
        cnn_kernel_sizes=[5, 3],
        cnn_activation_fn=nn.ELU(),
        classes=classes
    )

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=64,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(val_images, val_labels),
        batch_size=256,
    )

    train_model(
        model,
        train_loader,
        val_loader,
        epochs=20,
        lr=1e-3,
    )

    test = torch.load(test_file)

    test_images = test["depth_images"].unsqueeze(1).float()
    test_labels = torch.tensor(
        [class_to_idx[label] for label in test["labels"]],
        dtype=torch.long,
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=256,
    )

    evaluate(model, test_loader)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
        },
        folder / "depth_encoder_raw.pt",
    )

    example = torch.randn(
        1,
        1,
        *train_images.shape[-2:],   # (H, W)
    )

    scripted = torch.jit.script(model.cpu())
    scripted.save(str(folder / "depth_encoder_jit.pt"))

if __name__ == "__main__":
    main()
