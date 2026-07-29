import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import DepthTerrainClassifier
from legged_gym.utils.depth_terrain_classifier.bayesian_terrain_filter import (
    build_bayesian_filter_from_classifier,
    BayesianFilteredTerrainClassifier,
    search_filter_hyperparameters
)

calibration_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/train.pt"
fitted_model_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models/selector_model_99_86_20260725_200422.pt"


@contextmanager
def timed(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.2f} s")

def main():
    total_start = time.perf_counter()

    with timed("Setup"):
        fitted_model = DepthTerrainClassifier(pca_dim=4, num_prototypes=3)
        fitted_model.load(fitted_model_file)
        calibration = torch.load(calibration_file)

        calibration_labels, calibration_depth, calibration_rpy, calibration_omega = (
            calibration["labels"],
            calibration["depth_images"],
            calibration["orientation_rpy"],
            calibration["val_ang"]
        )

    with timed("Create validation model")
        bayes_filter, calibration_probs = build_bayesian_filter_from_classifier(
            fitted_model,
            validation_true_labels=calibration_labels,
            manual_prior={
                "rough": 0.70,
                "stairs": 0.10,
                "pit": 0.10,
                "gap": 0.10,
            },
            depth_images=calibration_depth,
            orientation_rpy=calibration_rpy,
            angular_velocity=calibration_omega,
            stay_probability=0.95,
            observation_mode="soft",
            observation_pseudocount=0.5,
            evidence_power=0.75,
            min_evidence_power=0.20,
            confidence_gamma=1.0,
        )

    validation_classifier_probs = calibration_probs
    validation_true_labels = calibration_labels

    with timed("Hyperparameter search"):
        results = search_filter_hyperparameters(
            validation_classifier_probs,
            validation_true_labels,
            labels=bayes_filter.labels,
            observation_matrix=bayes_filter.observation_matrix,
            manual_prior=bayes_filter.initial_prior,
            sequence_ids=None,
            stay_probabilities=[0.90, 0.94, 0.97],
            evidence_powers=[0.50, 0.75, 1.00],
            min_evidence_powers=[0.10, 0.25],
            confidence_gammas=[1.0, 2.0],
            scoring="balanced_accuracy",
        )

    print(results[0])

    best = results[0]

    with timed("Create optimized model"):
        bayes_filter, calibration_probs = build_bayesian_filter_from_classifier(
            fitted_model,
            validation_true_labels=calibration_labels,
            manual_prior={
                "rough": 0.70,
                "stairs": 0.10,
                "pit": 0.10,
                "gap": 0.10,
            },
            depth_images=calibration_depth,
            orientation_rpy=calibration_rpy,
            angular_velocity=calibration_omega,
            stay_probability=best["stay_probability"],
            observation_mode="soft",
            observation_pseudocount=0.5,
            evidence_power=best["evidence_power"],
            min_evidence_power=best["min_evidence_power"],
            confidence_gamma=best["confidence_gamma"],
        )
    
    del calibration_labels, calibration_depth, calibration_rpy, calibration_omega

    deployment_model = BayesianFilteredTerrainClassifier(
        fitted_model,
        bayes_filter,
    )

    test = torch.load(test_file)
    depth_images, orientation_rpy, angular_velocity, labels = (
        test["depth_images"],
        test["orientation_rpy"],
        test["angular_velocity"],
        test["labels"],
    )

    predictions_labels = []
    predictions_labels_inst = []

    for i in range(depth_images.shape[0]):
        if i % 1000 == 0:
            deployment_model.reset_temporal_filter()

        d = depth_images[i]
        o = orientation_rpy[i]
        a = angular_velocity[i]

        prediction = deployment_model.predict_depth(
            depth_image=d,
            orientation_rpy=o,
            angular_velocity=a,
        )

        predictions_labels.append(prediction.label)
        predictions_labels_inst.append(prediction.instantaneous_label)

    del depth_images, orientation_rpy, angular_velocity 

    correct_temporal = sum(
        pred == gt for pred, gt in zip(predictions_labels, labels)
    )
    correct_instantaneous = sum(
        pred == gt for pred, gt in zip(predictions_labels_inst, labels)
    )
    temporal_accuracy = 100 * correct_temporal / len(labels)
    instantaneous_accuracy = 100 * correct_instantaneous / len(labels)

    print(f"Temporal accuracy: {temporal_accuracy:.2f}% ({correct_temporal}/{len(labels)})")
    print(f"Instantaneous accuracy: {instantaneous_accuracy:.2f}% ({correct_instantaneous}/{len(labels)})")
    
    from collections import Counter

    print("Counting of labels", Counter(labels))
    print("Counting of predictions_labels", Counter(predictions_labels))
    print("Counting of predictions_labels_inst", Counter(predictions_labels_inst))

    del labels

    from datetime import datetime
    import os

    out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/complete_models"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(out_dir, f"baysian_{int(temporal_accuracy)}_intacc_{int(instantaneous_accuracy)}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    filter_path = os.path.join(model_dir, "bayes_filter.pt")
    import shutil

    print(f"Models saved to: {model_dir}")
    print(f"Total time {time.perf_counter() - total_start}")

    shutil.copy(fitted_model_file, model_dir)
    bayes_filter.save(filter_path)

if __name__ == "__main__":
    main()