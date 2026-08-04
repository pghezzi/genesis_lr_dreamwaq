from legged_gym import LEGGED_GYM_ROOT_DIR

import torch
import os


SEED = 42
FRAC = 0.1

def data_flattening(tensor):
    return tensor.transpose(0, 1).flatten(start_dim=0, end_dim=1)

def sample_envs(depth_images, base_rpy, base_ang_vel, fraction=0.5, seed=None):
    n_envs = depth_images.shape[1]
    n_keep = int(n_envs * fraction)

    g = None
    if seed is not None:
        g = torch.Generator().manual_seed(seed)

    idx = torch.randperm(n_envs, generator=g)[:n_keep]
    idx = idx.sort().values  # optional: preserve original ordering
    return (
        depth_images[:, idx],
        base_rpy[:, idx],
        base_ang_vel[:, idx],
    )

def reshape(tensor):
    shape = list(tensor.shape)
    shape[0] = shape[0] // 4
    shape[1] = shape[1] * 4
    return tensor.reshape(*shape)

def get_data_raw(load_file):
    torch_load = torch.load(load_file)
    filtered = torch_load["filtered"]
    depth_images = torch_load["depth_images"]
    base_rpy = torch_load["base_rpy"]
    base_ang_vel = torch_load["base_ang_vel"]
    terrain_name = torch_load["terrain_name"]
    if not filtered:
        depth_images = reshape(depth_images)
        base_rpy = reshape(base_rpy)
        base_ang_vel = reshape(base_ang_vel)
    depth_images, base_rpy, base_ang_vel = sample_envs(
        depth_images, base_rpy, base_ang_vel,
        fraction=FRAC,
        seed=SEED,
    )
    #train: [: 600], validate [600: 800], test [800: ]
    depth_images_flat = data_flattening(depth_images)
    base_rpy_flat = data_flattening(base_rpy)
    base_ang_vel_flat = data_flattening(base_ang_vel)

    return depth_images_flat, base_rpy_flat, base_ang_vel_flat, [terrain_name] * depth_images_flat.shape[0]

def get_data(load_file, skip=False):
    torch_load = torch.load(load_file)
    filtered = torch_load.get("filtered", False)
    depth_images = torch_load["depth_images"]
    base_rpy = torch_load["base_rpy"]
    base_ang_vel = torch_load["base_ang_vel"]
    episodes = depth_images.shape[0]
    envs = depth_images.shape[1]
    terrain_name = torch_load["terrain_name"]
    if not skip:
        if not filtered:
            depth_images = reshape(depth_images)
            base_rpy = reshape(base_rpy)
            base_ang_vel = reshape(base_ang_vel)
        
        depth_images, base_rpy, base_ang_vel = sample_envs(
            depth_images, base_rpy, base_ang_vel,
            fraction=FRAC,
            seed=SEED,
        )
    print(depth_images.shape[0], depth_images.shape[1])

    #torch.save(
    #    {
    #        "depth_images": depth_images,
    #        "depth_images_shape": tuple(depth_images.shape),
    #        "base_rpy": base_rpy,
    #        "base_rpy_shape": tuple(base_rpy.shape),
    #        "base_ang_vel": base_ang_vel,
    #        "base_ang_vel_shape": tuple(base_ang_vel.shape),
    #        "terrain_name": terrain_name,
    #        "filtered": True,
    #    },
    #    "custom_baseline_fix",
    #)
    #print("SHAPE: ", depth_images.shape[0])

    
    #train: [: 600], validate [600: 800], test [800: ]

    def _split_flat(t):
        train_split = int(t.shape[0] * 0.6)
        val_split = train_split + int(t.shape[0] * 0.2)
        return (
            data_flattening(t[:train_split]),
            data_flattening(t[train_split:val_split]),
            data_flattening(t[val_split:]),
        )

    train_depth, val_depth, test_depth = _split_flat(depth_images)
    train_rpy, val_rpy, test_rpy = _split_flat(base_rpy)
    train_ang, val_ang, test_ang = _split_flat(base_ang_vel)

    train = {
        "depth_images": train_depth,
        "orientation_rpy": train_rpy,
        "angular_velocity": train_ang,
        "labels": [terrain_name] * train_depth.shape[0],
    }

    val = {
        "depth_images": val_depth,
        "orientation_rpy": val_rpy,
        "angular_velocity": val_ang,
        "labels": [terrain_name] * val_depth.shape[0],
    }

    test = {
        "depth_images": test_depth,
        "orientation_rpy": test_rpy,
        "angular_velocity": test_ang,
        "labels": [terrain_name] * test_depth.shape[0],
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

path_calibration = f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_baseline_start_fresh/Jul07_06-24-55_dreamwaq_isaacgym/go2_depth_waq_baseline_start_fresh_plane_capture.pt"
path_baseline = f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_baseline_start_fresh/Jul07_06-24-55_dreamwaq_isaacgym/go2_depth_waq_baseline_start_fresh_baseline_capture.pt"
path_stairs = f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_stairs/Jul09_01-33-34_dreamwaq_genesis/go2_depth_waq_lora_8_stairs_stairs_capture.pt"
path_gap = f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_gap_experiment1_first_test/Jul16_19-41-34_dreamwaq_genesis/go2_depth_waq_lora_8_gap_experiment1_first_test_gap_capture.pt"

"""
python legged_gym/scripts/compile_depth_classifier_data.py --files * --calibration
"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile depth ang data")
    parser.add_argument("--files", nargs="+", help="files to compile")
    parser.add_argument("--calibration", type=str, default=None,
                        help="calibration file (optional)")
    parser.add_argument("--frac", type=float, default=0.1,
                        help="calibration file (optional)")
    parser.add_argument("--skip_reduce", type=int, default=None)

    args = parser.parse_args()
    FRAC = args.frac

    if args.calibration:
        calibration_depth, calibration_rpy, _, _ = get_data_raw(args.calibration)
    print(args.files)
    all_data    = [get_data(file, i == args.skip_reduce) for i, file in enumerate(args.files)]
    train       = merge_sets(*[data[0] for data in all_data])
    val         = merge_sets(*[data[1] for data in all_data])
    test        = merge_sets(*[data[2] for data in all_data])

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = f"{LEGGED_GYM_ROOT_DIR}/depth_waq_selector/processed_data/{timestamp}_frac_{str(FRAC).replace('.','_')}"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val,   os.path.join(out_dir, "val.pt"))
    torch.save(test,  os.path.join(out_dir, "test.pt"))

    if args.calibration:
        torch.save(
            {
                "depth_images": calibration_depth,
                "orientation_rpy": calibration_rpy,
            },
            os.path.join(out_dir, "calibration.pt"),
        )

    print(f"Saved train ({train['depth_images'].shape[0]} samples), "
          f"val ({val['depth_images'].shape[0]} samples), "
          f"test ({test['depth_images'].shape[0]} samples), "
          )
    if args.calibration:
        print(f"calibration ({calibration_depth.shape[0]} samples)")

    with open(os.path.join(out_dir,"parsed_arguments.txt"), "w") as f:
        f.write("Parsed Arguments\n")
        f.write("================\n")
        f.write(f"Files: {args.files}\n")
        f.write(f"Calibration: {args.calibration}\n")
        f.write(f"Frac: {args.frac}\n")
        f.write(f"Skip Reduce: {args.skip_reduce}\n")

    print(out_dir)

