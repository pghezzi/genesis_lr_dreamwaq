from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *

import sys
import time
import numpy as np
import torch
import pickle

# Compatibility shim: pickle files saved with NumPy >= 2.0 reference
# numpy._core (e.g. numpy._core.multiarray), which does not exist in
# NumPy < 2.0.  Register aliases so unpickling works transparently.
if not hasattr(np, '_core'):
    import numpy.core as _np_core
    for _attr in dir(_np_core):
        _mod_name = f"numpy._core.{_attr}"
        _submod = getattr(_np_core, _attr, None)
        if isinstance(_submod, type(sys)):
            sys.modules.setdefault(_mod_name, _submod)
    sys.modules.setdefault("numpy._core", _np_core)
    sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
    del _np_core, _attr, _mod_name, _submod

def main(args):
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1 # number of envs
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    
    # open the .pkl file and load the motion data
    # the motion data should be a dictionary with the following keys:
    # motion_data = {
    #     "fps": aligned_fps,
    #     "root_pos": root_pos,        # (N, 3), xyz world position of root
    #     "root_rot": root_rot,        # (N, 4), xyzw quaternion of root orientation
    #     "dof_pos": dof_pos,          # (N, num_dofs), joint angles matching dof_names order
    #     "local_body_pos": local_body_pos,  # (N, num_links, 3) or None
    #     "link_body_list": body_names,      # list of link names or None
    # }
    motion_file_dir = LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/"
    motion_file_path = motion_file_dir + env_cfg.env.motion_file
    with open(motion_file_path, "rb") as f:
        motion_data = pickle.load(f)
    aligned_fps = motion_data["fps"]
    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"]
    dof_pos = motion_data["dof_pos"]
    # convert np.array to torch tensor
    root_pos = torch.from_numpy(root_pos).to(env.device).float()
    root_rot = torch.from_numpy(root_rot).to(env.device).float()
    root_lin_vel = torch.zeros_like(root_pos)
    root_lin_vel[:-1] = (root_pos[1:] - root_pos[:-1]) * aligned_fps
    root_lin_vel[-1] = root_lin_vel[-2]  # set the last velocity to be the same as the second last one
    root_ang_vel = torch.zeros_like(root_pos)
    root_euler = get_euler_xyz(root_rot)
    root_ang_vel[:-1] = (root_euler[1:] - root_euler[:-1]) * aligned_fps
    root_ang_vel[-1] = root_ang_vel[-2]  # set the last velocity to be the same as the second last one
    dof_pos = torch.from_numpy(dof_pos).to(env.device).float()
    dof_vel = torch.zeros_like(dof_pos)
    dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) * aligned_fps
    dof_vel[-1] = dof_vel[-2]  # set the last velocity to be the same as the second last one
    all_indices = torch.arange(env.num_envs, device=env.device)
    
    num_frames = root_pos.shape[0]
    frame_dt = 1.0 / aligned_fps   # target wall-clock seconds per frame
    print(f"Playing {num_frames} frames at {aligned_fps} fps (frame_dt={frame_dt*1000:.1f} ms)")

    frame = 0
    key_body_pos_relative_to_base_list = []
    for i in range(num_frames):
        t_start = time.perf_counter()

        print(f"Loop frame {frame} / {num_frames}", end="\r")
        cur = i
        env.simulator.reset_dofs(all_indices,
                                 dof_pos[cur].unsqueeze(0),
                                 dof_vel[cur].unsqueeze(0))
        env.simulator.reset_root_states(all_indices,
                                        root_pos[cur].unsqueeze(0) + env.simulator.env_origins[all_indices],
                                        root_rot[cur].unsqueeze(0),
                                        root_lin_vel[cur].unsqueeze(0),
                                        root_ang_vel[cur].unsqueeze(0))
        # step the scene (not the env, to avoid torque control overriding dof positions)
        if SIMULATOR == "genesis":
            env.simulator._scene.step()
            cur_key_body_pos = env.simulator._robot.get_links_pos()[:, env.simulator._key_body_indices, :]
            cur_base_pos = env.simulator._robot.get_pos()
        elif SIMULATOR == "isaacgym":
            env.simulator._render()
            env.simulator._gym.simulate(env.simulator._sim)
            env.simulator._gym.fetch_results(env.simulator._sim, True)
            env.simulator._gym.refresh_actor_root_state_tensor(env.simulator._sim)
            env.simulator._gym.refresh_rigid_body_state_tensor(env.simulator._sim)
            cur_base_pos = env.simulator._root_states[:, 0:3]
            cur_key_body_pos = env.simulator._rigid_body_states[:, env.simulator._key_body_indices, 0:3]
        elif SIMULATOR == "isaaclab":
            env.simulator._sim.step(render=False)
            env.simulator._robot.update(env.simulator._sim_params["dt"])
            cur_key_body_pos = env.simulator._robot.data.body_link_pos_w[:, env.simulator._key_body_indices, :]
            cur_base_pos = env.simulator._robot.data.root_link_pos_w[:]
            env.simulator._sim.render()
        else:
            raise ValueError(f"Unsupported simulator: {SIMULATOR}")

        # record the caculated key body pos
        cur_key_body_pos_relative_to_base = cur_key_body_pos[0] - cur_base_pos[0].unsqueeze(0)
        key_body_pos_relative_to_base_list.append(cur_key_body_pos_relative_to_base)
        env.simulator.draw_debug_vis(cur_key_body_pos)
        
        # sleep for the remainder of the frame budget to match real-time playback
        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

        frame += 1
    
    # update the motion data with the recorded key body pos relative to base, and save to a new .pkl file
    key_body_pos_relative_to_base = torch.stack(key_body_pos_relative_to_base_list, dim=0)
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos.cpu().numpy(),
        "root_lin_vel": root_lin_vel.cpu().numpy(),
        "root_rot": root_rot.cpu().numpy(),
        "root_euler": root_euler.cpu().numpy(),
        "root_ang_vel": root_ang_vel.cpu().numpy(),
        "dof_pos": dof_pos.cpu().numpy(),
        "dof_vel": dof_vel.cpu().numpy(),
        "key_body_pos_relative_to_base": key_body_pos_relative_to_base.cpu().numpy(),
    }
    
    out_file_name = env_cfg.env.motion_file.replace(".pkl", f"_{SIMULATOR}.pkl")
    out_file_name = out_file_name.split("/")[-1]  # only keep the file name, remove the directory
    out_file_dir = args.motion_out_dir if args.motion_out_dir is not None else motion_file_dir
    output_file = os.path.join(LEGGED_GYM_ROOT_DIR, "resources/reference_motion", out_file_dir, out_file_name)
    with open(output_file, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"Saved updated motion data to {output_file}")

if __name__ == "__main__":
    args = get_args()
    main(args)
