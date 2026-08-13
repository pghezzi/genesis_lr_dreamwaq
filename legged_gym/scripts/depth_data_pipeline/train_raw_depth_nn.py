from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import NeuralClassifierAdapter
from util_func import fit_nn, evaluate_classifier, extract_in_chunks, save_classifier

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


        self.depth_image_resolution = tuple(depth_image_resolution)
        self.cnn_input_channel = cnn_input_channel
        self.cnn_channel_dims = list(cnn_channel_dims)
        self.cnn_strides = list(cnn_strides)
        self.cnn_fc_layer_dims = list(cnn_fc_layer_dims)
        self.cnn_kernel_sizes = list(cnn_kernel_sizes)
        self.cnn_activation_fn = cnn_activation_fn

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
        self.model = nn.Sequential(*cnn_layers)

    def forward(self, depth_image):
        return self.model(depth_image)
    
    def get_args(self) -> dict:
        """Return constructor kwargs sufficient to rebuild an equivalent (untrained)
        instance via `TerrainDepthClassifierNN(**model.get_args())`.

        Note: `cnn_activation_fn` is returned as the live module instance actually used
        (reused across all conv/fc layers already, so this matches original construction).
        For a JSON-serializable summary instead, use `cnn_activation_fn.__class__.__name__`.
        """
        return {
            "cls": self.__name__,
            "depth_image_resolution": tuple(self.depth_image_resolution),
            "cnn_input_channel": self.cnn_input_channel,
            "cnn_channel_dims": list(self.cnn_channel_dims),
            "cnn_strides": list(self.cnn_strides),
            "cnn_fc_layer_dims": list(self.cnn_fc_layer_dims),
            "cnn_kernel_sizes": list(self.cnn_kernel_sizes),
            "cnn_activation_fn": self.cnn_activation_fn,
        }

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

    

    train = torch.load(structural_training_data)
    structural_training_images = train["depth_images"].unsqueeze(1).float()
    structural_training_labels = train["labels"]

    validation = torch.load(validation_file)
    structural_validation_images = validation["depth_images"].unsqueeze(1).float()
    structural_validation_labels = validation["labels"]

    nn_raw_depth_model = TerrainDepthClassifierNN(
        depth_image_resolution=structural_training_images.shape[-2:],   # (H, W)
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

    classifier.fit(inputs=structural_training_images, labels=structural_training_labels, val=(structural_validation_images, structural_validation_labels), epochs=20)


    del train, structural_training_images, structural_training_labels
    del validation, structural_validation_images, structural_validation_labels

    test = torch.load(structural_test_data)
    structural_test_features = test["depth_images"].unsqueeze(1).float()
    structural_test_labels = test["labels"]

    acc = evaluate_classifier(classifier, structural_test_features, structural_test_labels)

    manual_prior = {
        label: 1.0 / len(classifier.class_ids)
        for label in classifier.class_ids
    }

    validation = torch.load(ordered_filter_validation)
    filter_validation_features = validation["depth_images"].unsqueeze(1).float()
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
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "depth_waq_selector", "full_models", f"nn_raw_depth_model_{timestamp}")
    bayes_filter.save(os.path.join(out_dir, "bayes_filter.pt"))
    classifier.save(os.path.join(out_dir, "classifier.pt"))
    torch.save(nn_raw_depth_model, os.path.join(out_dir, "nn_model_args.pt"))

if __name__ == "__main__":
    main()