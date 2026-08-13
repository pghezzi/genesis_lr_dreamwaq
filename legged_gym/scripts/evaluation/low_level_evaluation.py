"""
eval_policy.py

A simplified evaluation script for legged_gym trained policies.

Loads a trained checkpoint, runs it for a fixed number of simulation steps,
and reports:
  - Episode success rate  (episode reached max_episode_length without an
    early termination, e.g. the robot falling)
  - Velocity tracking error (commanded vs. actual linear/angular velocity)
  - Average distance covered per episode
"""

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *

import os
import json
import csv
import argparse
from datetime import datetime

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Device setup
# --------------------------------------------------------------------------- #

def _normalize_gpu_arg(gpu):
    gpu = str(gpu).strip().lower()
    if gpu.isdigit():
        return f"cuda:{gpu}"
    if gpu == "cuda":
        return gpu
    if gpu.startswith("cuda:"):
        index = gpu.split(":", 1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        f"Unsupported GPU specifier '{gpu}'. Use values like 'cuda', 'cuda:0', or '1'."
    )


def configure_runtime_device(args):
    """Normalize GPU selection and, when needed, mask visibility to the requested physical GPU."""
    if getattr(args, "cpu", False):
        args.gpu = "cpu"
        args.device = "cpu"
        return args

    requested_gpu = _normalize_gpu_arg(getattr(args, "gpu", "cuda:0"))
    runtime_gpu = requested_gpu

    if requested_gpu.startswith("cuda:"):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_gpu_ids = [g.strip() for g in visible_devices.split(",") if g.strip()]
            local_index = int(requested_gpu.split(":", 1)[1])
            if 0 <= local_index < len(visible_gpu_ids):
                runtime_gpu = f"cuda:{local_index}"
            else:
                physical_index = str(local_index)
                if physical_index not in visible_gpu_ids:
                    raise ValueError(
                        f"Requested GPU '{requested_gpu}' is not available under "
                        f"CUDA_VISIBLE_DEVICES={visible_devices}."
                    )
                runtime_gpu = f"cuda:{visible_gpu_ids.index(physical_index)}"
        else:
            physical_index = requested_gpu.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            runtime_gpu = "cuda:0"
    elif requested_gpu == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        runtime_gpu = "cuda:0"

    args.gpu = runtime_gpu
    args.device = runtime_gpu
    return args


def init_genesis(args, gs):
    """Initialize Genesis after device selection has been normalized."""
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu and args.gpu.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.gpu))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser(description="LeggedGym-Ex - Policy evaluation")
    parser.add_argument('--gpu', type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--task', type=str, default='go2', help="task name")
    parser.add_argument('--headless', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--num_envs', type=int, default=None, help="number of parallel environments")
    parser.add_argument('--load_run', type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt', type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--num_steps', type=int, default=1000, help="number of simulation steps to evaluate over")
    parser.add_argument('--save_path', type=str, default=None,
                         help="path to save results to (.json or .csv). If omitted, results are only printed.")
    return configure_runtime_device(parser.parse_args())


# --------------------------------------------------------------------------- #
# Env config
# --------------------------------------------------------------------------- #

def override_configs(env_cfg, args):
    """Minimal overrides needed for evaluation."""
    envs = args.num_envs if args.num_envs else min(env_cfg.env.num_envs, 100)
    env_cfg.env.num_envs = envs
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = env_cfg.env.num_envs
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    terrain_curriculum_difficulty["gap_size"]       = f"0.15 + 1.0 * difficulty"
    terrain_curriculum_difficulty["step_height"]    = f"0.10 + 0.3 * difficulty"
    terrain_curriculum_difficulty["pit_depth"]      = f"0.15 + 0.7 * difficulty"

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class EpisodeStats:
    """Tracks per-env running episode state and aggregates completed episodes."""

    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.ep_len = torch.zeros(num_envs, device=device)
        self.start_pos = None

        self.completed_lengths = []
        self.completed_success = []
        self.completed_distance = []
        self.vel_errors = []  # per-step (lin_err, ang_err) samples

    def step(self, base_pos, commands, actual_lin_vel, actual_ang_vel, dones, max_episode_length):
        if self.start_pos is None:
            self.start_pos = base_pos[:, :2].clone()

        self.ep_len += 1

        # Velocity tracking error: commanded vs. actual (xy linear speed + yaw rate)
        lin_err = torch.norm(commands[:, :2] - actual_lin_vel[:, :2], dim=-1)
        ang_err = torch.abs(commands[:, 2] - actual_ang_vel)
        self.vel_errors.append(torch.stack([lin_err, ang_err], dim=-1).cpu())

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            distance = torch.norm(base_pos[done_ids, :2] - self.start_pos[done_ids], dim=-1)
            lengths = self.ep_len[done_ids]
            # "success" = episode ran to (near) full length instead of terminating early
            success = lengths >= (max_episode_length - 1)

            self.completed_lengths.extend(lengths.cpu().tolist())
            self.completed_success.extend(success.cpu().tolist())
            self.completed_distance.extend(distance.cpu().tolist())

            self.ep_len[done_ids] = 0
            self.start_pos[done_ids] = base_pos[done_ids, :2]

    def summary(self):
        n = len(self.completed_lengths)
        vel_errors = torch.cat(self.vel_errors, dim=0) if self.vel_errors else torch.zeros(0, 2)
        return {
            "num_episodes": n,
            "success_rate": float(np.mean(self.completed_success)) if n else float("nan"),
            "avg_distance": float(np.mean(self.completed_distance)) if n else float("nan"),
            "avg_episode_length": float(np.mean(self.completed_lengths)) if n else float("nan"),
            "mean_lin_vel_error": float(vel_errors[:, 0].mean()) if vel_errors.numel() else float("nan"),
            "mean_ang_vel_error": float(vel_errors[:, 1].mean()) if vel_errors.numel() else float("nan"),
        }

    def save(self, path):
        """Save the summary + per-episode raw data to disk.

        Writes a JSON file at `path` (creating parent dirs as needed) containing
        the aggregate summary plus the raw per-episode lists (length, success,
        distance) so results can be re-analyzed later. If `path` ends in
        `.csv`, also writes a companion CSV with one row per completed episode
        (`<path>` itself, with a `.json` sibling for the summary).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

        summary = self.summary()
        payload = {
            "summary": summary,
            "episodes": {
                "length": self.completed_lengths,
                "success": self.completed_success,
                "distance": self.completed_distance,
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        if path.lower().endswith(".csv"):
            csv_path = path
            json_path = os.path.splitext(path)[0] + "_summary.json"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode_idx", "length", "success", "distance"])
                for i, (length, success, distance) in enumerate(
                    zip(self.completed_lengths, self.completed_success, self.completed_distance)
                ):
                    writer.writerow([i, length, int(success), distance])

            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)

            print(f"Saved per-episode data to: {csv_path}")
            print(f"Saved summary to:          {json_path}")
        else:
            json_path = path if path.lower().endswith(".json") else path + ".json"
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved evaluation results to: {json_path}")


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #

def evaluate(env, policy, args):
    obs = env.get_observations()
    if isinstance(obs, (tuple, list)):
        obs = obs[0]

    stats = EpisodeStats(env.num_envs, env.device)
    max_episode_length = env.max_episode_length

    for _ in range(args.num_steps):
        actions = policy(obs_buf.detach(), obs_history.detach(), depth.detach())
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos, depth = env.step(actions.detach())
        stats.step(
            base_pos=env.simulator.base_pos,
            commands=env.commands,
            actual_lin_vel=env.simulator.base_lin_vel,
            actual_ang_vel=env.simulator.base_ang_vel[:, 2],
            dones=dones,
            max_episode_length=max_episode_length,
        )
    return stats


def evaluate_policy(args):
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    if args.load_run:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.ckpt

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    stats = evaluate(env, policy, args)
    results = stats.summary()

    print("=== Evaluation Results ===")
    print(f"Episodes completed:     {results['num_episodes']}")
    print(f"Success rate:           {results['success_rate']:.3f}")
    print(f"Avg distance covered:   {results['avg_distance']:.3f} m")
    print(f"Avg episode length:     {results['avg_episode_length']:.1f} steps")
    print(f"Mean lin. vel. error:   {results['mean_lin_vel_error']:.4f} m/s")
    print(f"Mean ang. vel. error:   {results['mean_ang_vel_error']:.4f} rad/s")

    if args.save_path:
        stats.save(args.save_path)

    return results


if __name__ == '__main__':
    args = get_args()
    evaluate_policy(args)