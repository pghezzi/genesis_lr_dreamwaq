from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter
from util_func import fit_nn, evaluate_classifier, extract_in_chunks, make_terrain_extractor

import torch
import torch.nn as nn

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

def train_raw_depth_nn_from_data_set(train_file, test_file, validation_file, *_, **__):
    train = torch.load(train_file)
    validation = torch.load(validation_file)

    train_images = train["depth_images"].unsqueeze(1).float()
    train_labels = train["labels"]
    validation_images = validation["depth_images"].unsqueeze(1).float()
    validation_labels = validation["labels"]

    nn_raw_depth_model = TerrainDepthClassifierNN(
        depth_image_resolution=train_images.shape[-2:],   # (H, W)
        cnn_input_channel=1,                              # grayscale
        cnn_channel_dims=[8, 16],
        cnn_strides=[1, 1],
        cnn_fc_layer_dims=[128, len(set(train_labels))],
        cnn_kernel_sizes=[5, 3],
        cnn_activation_fn=nn.ELU(),
    )

    classifier = NeuralClassifierAdapter(
        model = nn_raw_depth_model,
        class_ids=[],
        input_transform=None,
        fit_callback=fit_nn,
    )

    classifier.fit(inputs=train_images, labels=train_labels, val=(validation_images, validation_labels), epochs=1)

    test = torch.load(test_file)
    test_features = test["depth_images"].unsqueeze(1).float()
    test_labels = test["labels"]

    acc = evaluate_classifier(classifier, test_features, test_labels)

    #out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models"
    #os.makedirs(out_dir, exist_ok=True)
    #
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #model_path = os.path.join(out_dir, f"raw_depth_classifier_acc_{str(acc).replace('.', '_')}_{timestamp}.pt")
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
    train_raw_depth_nn_from_data_set(train_file, test_file, validation_file)