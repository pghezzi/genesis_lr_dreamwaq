from __future__ import annotations

from .util_func import (
    collect_raw_depth_logits, json_safe, load_training_files,
    get_activation_fn, pack_raw_depth_state_inputs, processing_batch_size,
    save_results, sequence_ids_for,
    train_uncertainty_aware_nn_suite,
)

import torch
import torch.nn as nn
import time
from datetime import datetime
from pathlib import Path
from legged_gym import LEGGED_GYM_ROOT_DIR

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
        dropout_p=0.0,
        robot_state_dim=0,
    ):
        super().__init__()


        self.depth_image_resolution = tuple(depth_image_resolution)
        self.cnn_input_channel = cnn_input_channel
        self.cnn_channel_dims = list(cnn_channel_dims)
        self.cnn_strides = list(cnn_strides)
        self.cnn_fc_layer_dims = list(cnn_fc_layer_dims)
        self.cnn_kernel_sizes = list(cnn_kernel_sizes)
        self.cnn_activation_fn = get_activation_fn(cnn_activation_fn)
        self.dropout_p = float(dropout_p)
        self.robot_state_dim = int(robot_state_dim)
        if self.robot_state_dim < 0:
            raise ValueError("robot_state_dim must be non-negative")

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
                cnn_layers.append(self.cnn_activation_fn)

            in_channels = out_channels

            # Output size after conv
            in_height = (in_height - cnn_kernel_sizes[i]) // cnn_strides[i] + 1
            in_width = (in_width - cnn_kernel_sizes[i]) // cnn_strides[i] + 1

            # MaxPool after first conv
            if i == 0:
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                cnn_layers.append(self.cnn_activation_fn)

                in_height = (in_height - 2) // 2 + 1
                in_width = (in_width - 2) // 2 + 1

        cnn_layers.append(nn.Flatten())

        cnn_out_dim = in_height * in_width * cnn_channel_dims[-1]

        for l, dim in enumerate(cnn_fc_layer_dims):
            in_dim = cnn_out_dim + self.robot_state_dim if l == 0 else cnn_fc_layer_dims[l - 1]
            cnn_layers.append(nn.Linear(in_dim, dim))
            cnn_layers.append(self.cnn_activation_fn)
            if l != len(cnn_fc_layer_dims) - 1 and self.dropout_p > 0:
                cnn_layers.append(nn.Dropout(self.dropout_p))

        self._output_size = cnn_fc_layer_dims[-1]
        self.model = nn.Sequential(*cnn_layers)

    def forward(self, depth_image, robot_state=None):
        if self.robot_state_dim:
            if robot_state is None:
                expected_pixels = (self.cnn_input_channel
                                   * self.depth_image_resolution[0]
                                   * self.depth_image_resolution[1])
                if depth_image.ndim != 2 or depth_image.shape[1] != expected_pixels + self.robot_state_dim:
                    raise ValueError(
                        "state-aware raw-depth input must be packed as "
                        f"[B,{expected_pixels + self.robot_state_dim}]")
                robot_state = depth_image[:, expected_pixels:]
                depth_image = depth_image[:, :expected_pixels].reshape(
                    -1, self.cnn_input_channel, *self.depth_image_resolution)
            elif robot_state.shape[-1] != self.robot_state_dim:
                raise ValueError(f"robot_state must have {self.robot_state_dim} entries")
        elif robot_state is not None:
            raise ValueError("this model was constructed without robot-state inputs")

        value = depth_image
        state_pending = robot_state
        for layer in self.model:
            if state_pending is not None and isinstance(layer, nn.Linear):
                value = torch.cat((value, state_pending.to(value)), dim=1)
                state_pending = None
            value = layer(value)
        return value
    
    def get_args(self) -> dict:
        """Return constructor kwargs sufficient to rebuild an equivalent (untrained)
        instance via `TerrainDepthClassifierNN(**model.get_args())`.
        """
        return {
            "cls": self.__class__.__name__,
            "depth_image_resolution": tuple(self.depth_image_resolution),
            "cnn_input_channel": self.cnn_input_channel,
            "cnn_channel_dims": list(self.cnn_channel_dims),
            "cnn_strides": list(self.cnn_strides),
            "cnn_fc_layer_dims": list(self.cnn_fc_layer_dims),
            "cnn_kernel_sizes": list(self.cnn_kernel_sizes),
            "cnn_activation_fn": self.cnn_activation_fn.__class__.__name__,
            "dropout_p": self.dropout_p,
            "robot_state_dim": self.robot_state_dim,
        }

def main():
    args, files = load_training_files()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = args.output_dir or (Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "full_models" /
                                 f"raw_depth_nn_{datetime.now():%Y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=True)
    train = torch.load(files["train"], map_location="cpu", weights_only=False)
    structural_training_images = pack_raw_depth_state_inputs(
        train["depth_images"], train["orientation_rpy"], train["angular_velocity"])
    structural_training_labels = train["labels"]

    validation = torch.load(files["validation"], map_location="cpu", weights_only=False)
    structural_validation_images = pack_raw_depth_state_inputs(
        validation["depth_images"], validation["orientation_rpy"],
        validation["angular_velocity"])
    structural_validation_labels = validation["labels"]
    train_batch = processing_batch_size(args, len(structural_training_labels))
    validation_batch = processing_batch_size(args, len(structural_validation_labels))

    label_values = structural_training_labels.tolist() if torch.is_tensor(structural_training_labels) else list(structural_training_labels)
    class_ids = list(dict.fromkeys(label_values))
    split_data = {
        "structural_validation": validation,
        "structural_test": torch.load(files["structural_test"], map_location="cpu", weights_only=False),
        "ordered_validation": torch.load(files["bayes_validation"], map_location="cpu", weights_only=False),
        "ordered_test": torch.load(files["ordered_test"], map_location="cpu", weights_only=False),
    }
    def model_factory(dropout_p):
        return TerrainDepthClassifierNN(
            train["depth_images"].shape[-2:], 1, [8, 16], [1, 1],
            [128, len(class_ids)], [5, 3], nn.ELU(), dropout_p,
            robot_state_dim=5)
    def collect(classifier, data, samples, mc):
        return collect_raw_depth_logits(
            classifier, data, chunk_size=processing_batch_size(args, len(data["labels"])),
            mc_samples=samples, mc_dropout=mc)
    results = train_uncertainty_aware_nn_suite(
        architecture="raw_depth_nn", model_factory=model_factory, class_ids=class_ids,
        train_inputs=structural_training_images, train_labels=structural_training_labels,
        validation_inputs=structural_validation_images,
        validation_labels=structural_validation_labels, split_data=split_data,
        collect_logits=collect,
        validation_sequence_ids=sequence_ids_for(split_data["ordered_validation"]),
        test_sequence_ids=sequence_ids_for(split_data["ordered_test"]), output=output,
        device=device, train_batch=train_batch, validation_batch=validation_batch)
    results.update(method="raw-depth NN", require_feature=False, data_files=files,
                   robot_state_inputs=["roll", "pitch", "angular_velocity_roll",
                                       "angular_velocity_pitch", "angular_velocity_yaw"])
    save_results(output / "results.json", results)
    torch.save(json_safe(results), output / "results.pt")
    print(f"Saved raw-depth NN run to {output}")

if __name__ == "__main__":
    main()
