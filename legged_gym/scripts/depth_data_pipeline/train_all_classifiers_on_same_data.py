from train_raw_depth_nn import train_raw_depth_nn_from_data_set
from train_feature_nn import train_feature_nn_from_data_set
from train_rbf_prototype_classifier import train_rbf_prototype_from_data_set


func_to_train = [
    train_raw_depth_nn_from_data_set,
    train_feature_nn_from_data_set,
    train_rbf_prototype_from_data_set,
]

if __name__ == "__main__":
    from util_func import make_terrain_extractor
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
    extractor = make_terrain_extractor(files["calibration"])

    for func in func_to_train:
        func(
            train_file,
            test_file,
            validation_file,
            extractor
        )