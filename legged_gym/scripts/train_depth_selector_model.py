from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import DepthTerrainClassifier
import torch
import gc
from contextlib import contextmanager
import time

calibration_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/calibration.pt"
validation_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/val.pt"
train_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/train.pt"
test_file = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data/test.pt"

@contextmanager
def timed(name):
    start = time.perf_counter()
    yield
    print(f"{name}: {time.perf_counter() - start:.2f} s")

def main():
    total_start = time.perf_counter()
    

    with timed("Create validation model"):
        model = DepthTerrainClassifier(
            pca_dim = 8,
            num_prototypes=2,
            metric_type='euclidean',
            temperature=1.0,
            whiten=True,
            temporal_alpha=0.45,
            switch_frames=2,
            device='cpu'
        )
   
    with timed("Load calibration"):
        calibration = torch.load(calibration_file)
        calibration_depth, calibration_rpy = calibration["depth_images"], calibration["orientation_rpy"]
        
    with timed("Fit reference model"):
        model.fit_reference_model(calibration_depth, calibration_rpy)

    with timed("Hyperparameter search"):
        val = torch.load(validation_file)
        results = model.run_hyperparameter_search(
            **val,
            n_splits = 5,
            scoring="balanced_accuracy",
        )
        best = results[0]
        print(best)
        del model

    with timed("Create optimized model"):
        model = DepthTerrainClassifier(
            pca_dim = best["pca_dim"],
            num_prototypes= best["num_prototypes"],
            metric_type=best["metric_type"],
            metric_regularization=best["metric_regularization"],
            temperature=best["temperature"],
            whiten=best["whiten"],
            temporal_alpha=0.45,
            switch_frames=2,
        )    

    with timed("Fit reference model (best)"):
        model.fit_reference_model(calibration_depth, calibration_rpy)
        del calibration, calibration_depth, calibration_rpy

    

    with timed("Fit initial"):
        train = torch.load(train_file)
        model.fit_initial(
            **train
        )
        del train

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
            model.reset_temporal_filter()

        d = depth_images[i]
        o = orientation_rpy[i]
        a = angular_velocity[i]

        prediction = model.predict_depth(
            depth_image=d,
            orientation_rpy=o,
            angular_velocity=a,
        )

        predictions_labels.append(prediction.label)
        predictions_labels_inst.append(prediction.instantaneous_label)

    del depth_images, orientation_rpy, angular_velocity 


    # Compute accuracies

    #print(predictions_labels, predictions_labels_inst, labels)
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

    out_dir = "/home/pablo/Legged_Gym_EX/depth_waq_selector/models"
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, f"selector_model_{int(temporal_accuracy)}_{int(instantaneous_accuracy)}_{timestamp}.pt")

    model.save(model_path)
    print(f"Model saved to: {model_path}")
    print(f"Total time {time.perf_counter() - total_start}")

if __name__ == "__main__":
    model = main()
