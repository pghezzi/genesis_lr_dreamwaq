from legged_gym import LEGGED_GYM_ROOT_DIR

import torch
from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import DepthTerrainClassifier
from legged_gym.utils.depth_terrain_classifier.bayesian_terrain_filter import (
    build_bayesian_filter_from_classifier,
    BayesianFilteredTerrainClassifier,
    search_filter_hyperparameters
)
from contextlib import contextmanager
import time

#base_folder = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/20260803_115648_frac_0_2"
#base_folder = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/20260802_170851_frac_0_1"
#base_folder = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/99_85"
#base_folder = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/20260804_092752_frac_0_2"
base_folder = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/20260804_100953_frac_0_2"
calibration_file = f"{base_folder}/val.pt"
train_file = f"{base_folder}/train.pt"
test_file = f"{base_folder}/test.pt"
#fitted_model_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models/selector_model_99_86_20260725_200422.pt"
#fitted_model_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models/selector_model_20260802_170851_frac_0_1_acc_99_intacc_91_20260802_172350.pt"

#best 2
#fitted_model_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_20260803_132114_frac_0_1_acc_96_intacc_85_20260803_133617.pt"
#fitted_model_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_20260802_170851_frac_0_1_acc_99_intacc_91_20260802_172350.pt"


#fitted_model_file =  "/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_20260803_203406_frac_0_1_acc_97_intacc_86_20260803_205340.pt"

#fitted_model_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_20260803_222015_frac_0_1_acc_99_intacc_90_20260803_223733.pt"

fitted_model_file = f"/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_20260804_094928_frac_0_1_acc_92_intacc_90_20260804_100715.pt"

#calibration_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/99_85/val.pt"
#train_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/99_85/train.pt"
#test_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/99_85/test.pt"
##fitted_model_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models/selector_model_20260801_001436_frac_0_1_acc_99_intacc_87_20260801_003007.pt"
#fitted_model_file = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/models/selector_model_20260802_100447_frac_0_2_acc_95_intacc_80_20260802_101929.pt"


#calibration_file = "depth_waq_selector/processed_data/20260731_153919_frac_0_1/val.pt"
#train_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/20260731_153919_frac_0_1/train.pt"
#test_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/20260731_153919_frac_0_1/test.pt"
#fitted_model_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/models/selector_model_99_86_20260725_200422.pt"

@contextmanager
def timed(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.2f} s")

def main():
    total_start = time.perf_counter()

    with timed("Setup"):

        print(f"Using data from: {base_folder}")
        print(f"Using fitted model: {fitted_model_file}")
        manual_prior = {
                "baseline": 0.80,
                "stairs": 0.10,
                #"pit": 0.10,
                "gap": 0.10,
            }
        fitted_model = DepthTerrainClassifier(pca_dim=4, num_prototypes=3)
        fitted_model.reset_temporal_filter()
        fitted_model.load(fitted_model_file)
        calibration = torch.load(train_file)

        calibration_labels, calibration_depth, calibration_rpy, calibration_omega, calibration_seq = (
            calibration["labels"],
            calibration["depth_images"],
            calibration["orientation_rpy"],
            calibration["angular_velocity"],
            calibration["seq"]
        )

    with timed("Create validation model"):
        bayes_filter, calibration_probs = build_bayesian_filter_from_classifier(
            fitted_model,
            validation_true_labels=calibration_labels,
            manual_prior=manual_prior,
            depth_images=calibration_depth,
            orientation_rpy=calibration_rpy,
            angular_velocity=calibration_omega,
            stay_probability=0.975,
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
            sequence_ids=calibration_seq,
            stay_probabilities=[ 0.95, 0.96, 0.97, 0.98, 0.99 ],
            evidence_powers=[0.0, 0.25, 0.50, 0.75, 1.00],
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
            manual_prior=bayes_filter.initial_prior,
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
    
    del calibration_labels, calibration_depth, calibration_rpy, calibration_omega, calibration_seq

    deployment_model = BayesianFilteredTerrainClassifier(
        fitted_model,
        bayes_filter,
    )

    test = torch.load(test_file)
    depth_images, orientation_rpy, angular_velocity, labels, seq = (
        test["depth_images"],
        test["orientation_rpy"],
        test["angular_velocity"],
        test["labels"],
        test["seq"]
    )

    predictions_labels = []
    predictions_labels_inst = []

    for i in range(depth_images.shape[0]):
        if i == 0 or seq[i] != seq[i-1]:
            deployment_model.reset()
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

    del depth_images, orientation_rpy, angular_velocity, seq

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
    import shutil

    print(f"Models saved to: {model_dir}")
    print(f"Total time {time.perf_counter() - total_start}")

    shutil.copy(fitted_model_file, model_dir)
    fitted_model.reset_temporal_filter()
    bayes_filter.reset()
    fitted_model.save(os.path.join(model_dir, "fitted_model.pt"))
    bayes_filter.save(os.path.join(model_dir, "bayes_filter.pt"))

    torch.save(
        {
            "pca_dim":          fitted_model.classifier.pca_dim,
            "num_prototypes":   fitted_model.classifier.num_prototypes,
            "lables":           bayes_filter.labels,
            "prior":            bayes_filter.initial_prior,
            "transition" :      bayes_filter.transition_matrix,
            "observation" :     bayes_filter.observation_matrix
        }, os.path.join(model_dir, "args.pt")
    )

if __name__ == "__main__":
    main()