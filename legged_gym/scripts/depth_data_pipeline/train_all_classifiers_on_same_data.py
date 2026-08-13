from legged_gym import LEGGED_GYM_ROOT_DIR

from train_raw_depth_nn import train_raw_depth_nn_from_data_set
from train_feature_nn import train_feature_nn_from_data_set
from train_rbf_prototype_classifier import train_rbf_prototype_from_data_set

from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        ProbabilisticClassifier,
        PCAWhitenedRBFPrototypeClassifier,
        NeuralClassifierAdapter
    )


func_to_train = {
    "raw_depth_nn": train_raw_depth_nn_from_data_set,
    "feature_nn": train_feature_nn_from_data_set,
    "rbf_prototype": train_rbf_prototype_from_data_set,
}

if __name__ == "__main__":
    from util_func import make_terrain_extractor
    import argparse
    from datetime import datetime
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Create model using data")
    parser.add_argument("--folder", type=str, default=None, help="folder with data")
    args = parser.parse_args() 
    folder = Path(args.folder)
    files = { name : path for name in ("calibration", "val", "train", "test") if (path := folder / f"{name}.pt").is_file() } 
    train_file = files["train"] 
    test_file = files["test"]
    validation_file = files["val"]
    extractor = make_terrain_extractor(files["calibration"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
    out_dir = Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "classifiers" / f"classifiers_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    for name, func in func_to_train.items():
        classifier = func(
            train_file,
            test_file,
            validation_file,
            extractor
        )
        classifier_dir = save_classifier(classifier, files, extractor)

        print(f"Saved classifier to: {classifier_dir}")
        