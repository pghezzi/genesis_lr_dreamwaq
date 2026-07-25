import torch
import os


SEED = 42
FRAC = 0.1

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

#future, move to real
def reshape(tensor):
    shape = list(tensor.shape)
    shape[0] = shape[0] // 4
    shape[1] = shape[1] * 4
    return tensor.reshape(*shape)

def get_data_raw(load_file):
    torch_load = torch.load(load_file)
    depth_images = reshape(torch_load["depth_images"])
    base_rpy = reshape(torch_load["base_rpy"])
    base_ang_vel = reshape(torch_load["base_ang_vel"])
    terrain_name = torch_load["terrain_name"]

    depth_images, base_rpy, base_ang_vel = sample_envs(
        depth_images, base_rpy, base_ang_vel,
        fraction=FRAC,
        seed=SEED,
    )
    #train: [: 600], validate [600: 800], test [800: ]
    depth_images_flat = torch.flatten(depth_images, start_dim=0, end_dim=1)
    base_rpy_flat = torch.flatten(base_rpy, start_dim=0, end_dim=1)
    base_ang_vel_flat = torch.flatten(base_ang_vel, start_dim=0, end_dim=1)

    return depth_images_flat, base_rpy_flat, base_ang_vel_flat, [terrain_name] * depth_images_flat.shape[0]

def get_data(load_file):
    torch_load = torch.load(load_file)
    depth_images = reshape(torch_load["depth_images"])
    base_rpy = reshape(torch_load["base_rpy"])
    base_ang_vel = reshape(torch_load["base_ang_vel"])
    episodes = depth_images.shape[0]
    envs = depth_images.shape[1]
    terrain_name = torch_load["terrain_name"]

    depth_images, base_rpy, base_ang_vel = sample_envs(
        depth_images, base_rpy, base_ang_vel,
        fraction=FRAC,
        seed=SEED,
    )
    #train: [: 600], validate [600: 800], test [800: ]

    def _split_flat(t):
        return (
            torch.flatten(t[:600], start_dim=0, end_dim=1),
            torch.flatten(t[600:800], start_dim=0, end_dim=1),
            torch.flatten(t[800:], start_dim=0, end_dim=1),
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

path_plane = "/home/pablo/Legged_Gym_EX/logs/go2_depth_waq_baseline_start_fresh/Jul07_06-24-55_dreamwaq_isaacgym/go2_depth_waq_baseline_start_fresh_plane_capture.pt"
    
path_baseline = "/home/pablo/Legged_Gym_EX/logs/go2_depth_waq_baseline_start_fresh/Jul07_06-24-55_dreamwaq_isaacgym/go2_depth_waq_baseline_start_fresh_baseline_capture.pt"
path_stairs = "/home/pablo/Legged_Gym_EX/logs/go2_depth_waq_lora_8_stairs/Jul09_01-33-34_dreamwaq_genesis/go2_depth_waq_lora_8_stairs_stairs_capture.pt"
path_gap = "/home/pablo/Legged_Gym_EX/logs/go2_depth_waq_lora_8_gap_experiment1_first_test/Jul16_19-41-34_dreamwaq_genesis/go2_depth_waq_lora_8_gap_experiment1_first_test_gap_capture.pt"



if __name__ == "__main__":    
    calibration_depth, calibration_rpy, _, _ = get_data_raw(path_plane)
    all_data    = [get_data(file) for file in (path_baseline, path_stairs, path_gap)]
    train       = merge_sets(*[data[0] for data in all_data])
    val         = merge_sets(*[data[1] for data in all_data])
    test        = merge_sets(*[data[2] for data in all_data])

    out_dir = "/home/pablo/Legged_Gym_EX/depth_waq_selector/processed_data"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(train, os.path.join(out_dir, "train.pt"))
    torch.save(val,   os.path.join(out_dir, "val.pt"))
    torch.save(test,  os.path.join(out_dir, "test.pt"))

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
          f"calibration ({calibration_depth.shape[0]} samples) to {out_dir}")

