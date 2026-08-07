from legged_gym import LEGGED_GYM_ROOT_DIR

import torch
import os

def data_flattening(tensor):
    #data is stacked as episodes, envs, shape. This swaps it to ensure 
    return tensor.transpose(0, 1).flatten(start_dim=0, end_dim=1)

def labels_flattening(labels):
    # labels: [episodes][envs] -> [envs * episodes]
    return [label for env in zip(*labels) for label in env]

def sample_envs(depth_images, base_rpy, base_ang_vel, terrain_labels, fraction=0.5, seed=None):
    n_envs = depth_images.shape[1]
    n_keep = int(n_envs * fraction)

    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)

    idx = torch.randperm(n_envs, generator=g)[:n_keep]
    idx = idx.sort().values  # optional: preserve original ordering

    terrain_labels = [
        [row[i] for i in idx.tolist()]
        for row in terrain_labels
    ]

    return (
        depth_images[:, idx],
        base_rpy[:, idx],
        base_ang_vel[:, idx],
        terrain_labels
    )

def get_data_raw(load_file, frac=0.1, seed=42):
    torch_load = torch.load(load_file)
    depth_images = torch_load["depth_images"]
    base_rpy = torch_load["base_rpy"]
    base_ang_vel = torch_load["base_ang_vel"]
    terrain_labels = torch_load["terrain_name"]
    filtered = torch_load["filtered"]
    depth_images, base_rpy, base_ang_vel, terrain_labels = sample_envs(
        depth_images, base_rpy, base_ang_vel, terrain_labels,
        fraction=frac,
        seed=seed,
    )
    return depth_images, base_rpy, base_ang_vel, terrain_labels

def calibration_data(*args, **kwargs):
    calibration_depth, calibration_rpy, _, _  = get_data_raw(*args, **kwargs)
    return {
        "depth_images": data_flattening(calibration_depth),
        "orientation_rpy": data_flattening(calibration_rpy),
    }

def train_val_test_data(*args, **kwargs):
    depth_images, base_rpy, base_ang_vel, terrain_labels = get_data_raw(*args, **kwargs)

    def _split_flat(t):
        train_split = int(t.shape[0] * 0.6)
        val_split = train_split + int(t.shape[0] * 0.2)
        return (
            data_flattening(t[:train_split]),
            data_flattening(t[train_split:val_split]),
            data_flattening(t[val_split:]),
        )
    
    def _split_list(lst):
        train_split = int(len(lst) * 0.6)
        val_split = train_split + int(len(lst) * 0.2)
        return (
            labels_flattening(lst[:train_split]),
            labels_flattening(lst[train_split:val_split]),
            labels_flattening(lst[val_split:]),
        )
    
    train_depth, val_depth, test_depth = _split_flat(depth_images)
    train_rpy, val_rpy, test_rpy = _split_flat(base_rpy)
    train_ang, val_ang, test_ang = _split_flat(base_ang_vel)
    train_labels, val_labels, test_labels = _split_list(terrain_labels)

    assert len(train_labels) == train_depth.shape[0]

    train = {
        "depth_images": train_depth,
        "orientation_rpy": train_rpy,
        "angular_velocity": train_ang,
        "labels": train_labels,
    }

    val = {
        "depth_images": val_depth,
        "orientation_rpy": val_rpy,
        "angular_velocity": val_ang,
        "labels": val_labels,
    }

    test = {
        "depth_images": test_depth,
        "orientation_rpy": test_rpy,
        "angular_velocity": test_ang,
        "labels": test_labels,
    }

    return train, val, test

def merge_sets(*datasets):
    merged = {}
    for key in datasets[0]:
        if key == "labels":
            merged[key] = sum((d[key] for d in datasets), [])
        else:
            merged[key] = torch.cat([d[key] for d in datasets], dim=0)
    return merged

# reshaped removed as this is now handled by the data generator

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Compile depth ang data")
    parser.add_argument("--files", nargs="+", help="files to compile")
    parser.add_argument("--calibration", type=str, default=None,
                        help="calibration file (optional)")
    parser.add_argument("--frac", type=float, default=0.1,
                        help="amount of envs used from total(optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for random events(optional)")

    args = parser.parse_args()

    get_data_args = {
        "frac": args.frac,
        "seed": args.seed
    }

    if args.calibration:
        calibration = calibration_data(args.calibration, **get_data_args)
    
    all_data    = [train_val_test_data(file, **get_data_args) for i, file in enumerate(args.files)]
    train       = merge_sets(*[data[0] for data in all_data])
    val         = merge_sets(*[data[1] for data in all_data])
    test        = merge_sets(*[data[2] for data in all_data])

    out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/{timestamp}_frac_{str(args.frac).replace('.','_')}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir,"parsed_arguments.txt"), "w") as f:
        f.write("Parsed Arguments\n")
        f.write("================\n")
        f.write(f"Files: {args.files}\n")
        f.write(f"Calibration: {args.calibration}\n")
        f.write(f"Frac: {args.frac}\n")
    
    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val,   os.path.join(out_dir, "val.pt"))
    torch.save(test,  os.path.join(out_dir, "test.pt"))
    if args.calibration:
        torch.save(calibration,  os.path.join(out_dir, "calibration.pt"))

    print(out_dir)
    

