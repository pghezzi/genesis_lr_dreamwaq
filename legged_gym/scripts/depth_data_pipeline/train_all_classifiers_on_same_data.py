from legged_gym import LEGGED_GYM_ROOT_DIR

from train_raw_depth_nn import train_raw_depth_nn_from_data_set
from train_feature_nn import train_feature_nn_from_data_set
from train_rbf_prototype_classifier import train_rbf_prototype_from_data_set


func_to_train = {
    "raw_depth_nn":train_raw_depth_nn_from_data_set,
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

    out_dir = Path(LEGGED_GYM_ROOT_DIR) / "depth_waq_selector" / "classifiers"
    os.makedirs(out_dir, exist_ok=True)

    for name, func in func_to_train.items():
        classifier = func(
            train_file,
            test_file,
            validation_file,
            extractor
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        classifier_dir = out_dir / f"{name}_{timestamp}"
        classifier_dir.mkdir(parents=True, exist_ok=True) # Save classifier classifier.save(classifier_dir / "classifier.pt") 
        # Save the exact data files used to create this classifier 
        #for file_name, source_path in files.items(): 
        #    shutil.copy2( source_path, classifier_dir / f"{file_name}.pt", ) 
        # TOO BIG
        # Save metadata, including the original data folder
        metadata = {
            "classifier_name": name,
            "timestamp": timestamp,
            "data_folder": str(folder.resolve()),
            "data_files": {
                file_name: str(source_path.resolve())
                for file_name, source_path in files.items()
            },
        }

        with open(classifier_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved classifier to: {classifier_dir}")
        