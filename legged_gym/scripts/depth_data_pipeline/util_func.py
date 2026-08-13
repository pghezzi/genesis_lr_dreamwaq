from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import PCAWhitenedRBFPrototypeClassifier, NeuralClassifierAdapter
from .train_feature_nn import TerrainDepthFeatureClassifierNN
from .train_raw_depth_nn import TerrainDepthClassifierNN
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import os
from util_func import make_terrain_extractor
import argparse
from datetime import datetime
from pathlib import Path

def make_terrain_extractor(calibration_file):
    calibration = torch.load(calibration_file)
    calibration_images = calibration["depth_images"].float()
    calibration_rpy = calibration["orientation_rpy"]
    extractor = SobelDepthTerrainFeatureExtractor(
        output_size=(calibration_images.shape[-2:]),
        min_depth=0.02,
        max_depth=1.0,
        far_depth=0.6,
        close_depth=0.15,
        close_residual_threshold=0.05,
        sobel_edge_threshold=0.007,
        depth_scale=None
    )
    extractor.fit_reference_model(calibration_images, calibration_rpy)
    del calibration_images, calibration_rpy
    return extractor

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

def extract_in_chunks(extractor, depth_images, orientation_rpy, angular_velocity, chunk_size=256, device="cpu"):
    n = depth_images.shape[0]
    outputs = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Only convert/cast the slice, not the whole tensor
        depth_chunk = depth_images[start:end].float()
        rpy_chunk = orientation_rpy[start:end]
        ang_vel_chunk = angular_velocity[start:end]

        with torch.no_grad():
            feats = extractor.extract_batch(depth_chunk, rpy_chunk, ang_vel_chunk)

        outputs.append(feats.detach().to(device))

        # drop references to the chunk explicitly
        del depth_chunk, rpy_chunk, ang_vel_chunk, feats

    return torch.cat(outputs, dim=0)


def evaluate_classifier(classifier, test_inputs, test_labels):
    instantaneous_metrics = classifier.evaluate(
        test_inputs,
        test_labels,
        temperature=1.0,
    )

    print("Instantaneous accuracy:", instantaneous_metrics["accuracy"])
    print(
        "Instantaneous balanced accuracy:",
        instantaneous_metrics["balanced_accuracy"],
    )
    print("Instantaneous macro F1:", instantaneous_metrics["macro_f1"])
    print("Confusion matrix:\n", instantaneous_metrics["confusion_matrix"])
    return instantaneous_metrics["accuracy"]


def save_classifier(classifier, files, extractor = None):
    out_dir = Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "classifiers"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    classifier_dir = out_dir / f"{name}_{timestamp}"
    classifier_dir.mkdir(parents=True, exist_ok=True) # Save classifier classifier.save(classifier_dir / "classifier.pt")
    classifier.save(classifier_dir / f"{name}_classifier")
    if extractor and classifier.require_feature:
        extractor.save(classifier_dir / f"{name}_extractor")
    metadata = {
        "class": classifier.__name__,
        "timestamp": timestamp,
        "data_folder": str(folder.resolve()),
        "data_files": {
            file_name: str(source_path.resolve())
            for file_name, source_path in files.items()
        },
    }

    if hasattr(classifier, "model"):
        metadata["nn_model"] = classifier.model.__name__
        metadata["nn_model_args"] = classifier.model.get_args()

    with open(classifier_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return classifier_dir

def _find_one(classifier_dir: Path, pattern: str) -> Optional[Path]:
    """Return the single file matching `pattern` in `classifier_dir`, or None."""
    matches = sorted(classifier_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching '{pattern}' in {classifier_dir}, "
            f"found {[m.name for m in matches]}"
        )
    return matches[0]

def load_classifier_extractor(
    classifier_dir: str | Path,
) -> tuple[Any, Any, dict]:
    classifier_dir = Path(classifier_dir)

    metadata_path = classifier_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {classifier_dir}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    class_name = metadata["class"]
    classifier_path = _find_one(classifier_dir, "*_classifier*")
    extractor_path = _find_one(classifier_dir, "*_extractor*")

    if classifier_path is None:
        raise FileNotFoundError(f"No classifier checkpoint found in {classifier_dir}")

    if "nn_model" in metadata:
        model_class = metadata["nn_model"]
        model_args = metadata["nn_model_args"]
        model_blank = eval(model_class)(**model_args)
        classifier = eval(class_name).load(classifier_path, model_blank)
    else:
        classifier = eval(class_name).load(classifier_path)
    
    if extractor_path:
        extractor = SobelDepthTerrainFeatureExtractor.load(extractor_path)
    else:
        extractor = None

    return classifier, extractor
