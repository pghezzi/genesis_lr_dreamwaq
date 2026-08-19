"""Evaluate one low-level policy over a global terrain-difficulty curriculum."""

# TERRAIN is read while task modules are imported, so select it first.
import os
import sys

SKILL_TO_TERRAIN = {
    "rough": "baseline",
    "leap": "gap",
    "climb": "pit",
    "stairs": "stairs",
}


def _bootstrap_terrain_from_cli():
    skill = "rough"
    arguments = sys.argv[1:]
    for index, argument in enumerate(arguments):
        if argument == "--skill" and index + 1 < len(arguments):
            skill = arguments[index + 1]
            break
        if argument.startswith("--skill="):
            skill = argument.split("=", 1)[1]
            break
    if skill in SKILL_TO_TERRAIN:
        os.environ["TERRAIN"] = SKILL_TO_TERRAIN[skill]


_bootstrap_terrain_from_cli()

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *

import argparse
import csv
import json
import random
from datetime import datetime

import numpy as np
import torch

DEFAULT_COMMANDS = {"rough": 0.8, "leap": 1.5, "climb": 1.2, "stairs": 1.2}
DIFFICULTY_RANGES = {
    "leap": (0.30, 1.00, "gap_size"),
    "climb": (0.25, 0.50, "pit_depth"),
    "stairs": (0.10, 0.40, "step_height"),
}

# Explicit evaluation ranges, kept aligned with the depth-WAQ training config.
EVAL_DOMAIN_RANDOMIZATION_RANGES = {
    "friction_range": [1.0, 1.0],
    "com_pos_x_range": [-0.0, 0.0],
    "com_pos_y_range": [-0.0, 0.0],
    "com_pos_z_range": [-0.0, 0.0],
    "kp_range": [1.0, 1.0],
    "kd_range": [1.0, 1.0],
    "motor_strength_range": [1.0, 1.0],
    "ctrl_delay_step_range": [0, 0],
    "joint_armature_range": [0.020, 0.020],
    "joint_friction_range": [0.015, 0.015],
    "joint_damping_range": [0.275, 0.275],
    "camera_com_displacement_range": [0.0, 0.0, 0.0],
    "camera_euler_offset_range": [0.0, 0.0, 0.0],
}
EVAL_DOMAIN_RANDOMIZATION_ENABLED = {
    "randomize_friction": True,
    "randomize_restitution": False,
    "randomize_base_mass": False,
    "randomize_com_displacement": False,
    "randomize_ctrl_delay": False,
    "randomize_pd_gain": False,
    "randomize_motor_strength": False,
    "randomize_joint_armature": True,
    "randomize_joint_friction": True,
    "randomize_joint_damping": True,
    "push_robots": False,
    "push_links": False,
    "randomize_camera_pos": False,
    "randomize_camera_euler": False,
}


def _normalize_gpu_arg(gpu):
    gpu = str(gpu).strip().lower()
    if gpu.isdigit():
        return f"cuda:{gpu}"
    if gpu == "cuda":
        return gpu
    if gpu.startswith("cuda:") and gpu.split(":", 1)[1].isdigit():
        return gpu
    raise ValueError(
        f"Unsupported GPU specifier '{gpu}'. Use values like 'cuda', 'cuda:0', or '1'."
    )


def configure_runtime_device(args):
    """Preserve the evaluator's existing physical/local CUDA-device handling."""
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
            elif str(local_index) in visible_gpu_ids:
                runtime_gpu = f"cuda:{visible_gpu_ids.index(str(local_index))}"
            else:
                raise ValueError(
                    f"Requested GPU '{requested_gpu}' is not available under "
                    f"CUDA_VISIBLE_DEVICES={visible_devices}."
                )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = requested_gpu.split(":", 1)[1]
            runtime_gpu = "cuda:0"
    elif requested_gpu == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        runtime_gpu = "cuda:0"

    args.gpu = runtime_gpu
    args.device = runtime_gpu
    return args


def init_genesis(args, gs):
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu and args.gpu.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.gpu))


def get_args():
    parser = argparse.ArgumentParser(description="LeggedGym-Ex low-level policy evaluation")
    parser.add_argument("--gpu", type=str, default="cuda:0", help="GPU, e.g. cuda:0 or 1")
    parser.add_argument("--task", type=str, default="go2_depth_waq_lora", help="registered task name")
    parser.add_argument("--skill", choices=tuple(SKILL_TO_TERRAIN), default="rough")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False, help="use CPU")
    parser.add_argument("--num_envs", type=int, default=None, help="parallel environments")
    parser.add_argument("--load_run", type=str, default=None, help="run to load; defaults to latest")
    parser.add_argument("--ckpt", type=int, default=-1, help="checkpoint; -1 means latest")
    parser.add_argument("--num_difficulty_levels", type=int, default=10)
    parser.add_argument("--episodes_per_level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--success_distance", type=float, default=None,
        help="forward +x meters; defaults to the configured sub-terrain length",
    )
    parser.add_argument(
        "--forward_command", type=float, default=None,
        help="fixed forward command; defaults to the skill's training curriculum cap",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100000,
        help="global simulation-step safety cap (episode quotas normally stop evaluation)",
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="results path (.json or .csv); omitted means print only",
    )
    args = parser.parse_args()
    if args.num_difficulty_levels < 1 or args.episodes_per_level < 1:
        parser.error("difficulty levels and episodes per level must be positive")
    if (args.success_distance is not None and args.success_distance <= 0) or args.num_steps < 1:
        parser.error("success distance and num steps must be positive")
    return configure_runtime_device(args)


def _difficulty_expression(start, end, levels):
    """Compensate for terrain generation using difficulty=row/num_rows."""
    if levels == 1:
        return f"{start:.12g}"
    slope = (end - start) * levels / (levels - 1)
    return f"{start:.12g} + {slope:.12g} * difficulty"


def override_configs(env_cfg, args):
    """Apply evaluation-only terrain, reset, and straight-command settings."""
    env_cfg.env.num_envs = args.num_envs or min(env_cfg.env.num_envs, 100)
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = env_cfg.env.num_envs
    if args.success_distance is None:
        args.success_distance = float(env_cfg.terrain.terrain_length)

    env_cfg.terrain.num_rows = args.num_difficulty_levels
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.curriculum = False
    if args.skill in DIFFICULTY_RANGES:
        start, end, parameter = DIFFICULTY_RANGES[args.skill]
        env_cfg.terrain.terrain_curriculum_difficulty[parameter] = _difficulty_expression(
            start, end, args.num_difficulty_levels
        )

    command = args.forward_command
    if command is None:
        command = DEFAULT_COMMANDS[args.skill]
    args.forward_command = float(command)
    env_cfg.init_state.yaw_random_scale = 0.0
    env_cfg.commands.curriculum = False
    if hasattr(env_cfg.commands, "custom_command_curriculum"):
        env_cfg.commands.custom_command_curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.ranges.lin_vel_x = [command, command]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    # Do not depend on inherited training ranges during evaluation.
    for name, value in EVAL_DOMAIN_RANDOMIZATION_RANGES.items():
        setattr(env_cfg.domain_rand, name, list(value))
    if args.skill == "rough":
        env_cfg.domain_rand.added_mass_range = [-1.0, 1.0]
        env_cfg.domain_rand.push_interval_s = 10
        env_cfg.domain_rand.max_push_vel_xy = 1.0
    else:
        env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]
        env_cfg.domain_rand.push_interval_s = 3
        env_cfg.domain_rand.max_push_vel_xy = 0.5

    for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items():
        setattr(env_cfg.domain_rand, name, enabled)


def _unpack_observations(observations):
    if not isinstance(observations, (tuple, list)) or len(observations) < 6:
        raise RuntimeError(
            "This evaluator expects depth-WAQ observations: "
            "(obs, privileged_obs, obs_history, explicit_labels, next_state, depth)."
        )
    return observations[0], observations[2], observations[5]


def _force_commands(env, forward_command):
    env.commands[:, 0] = forward_command
    env.commands[:, 1:3] = 0.0
    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0


def _set_global_level(env, level):
    """Assign the same terrain row to every environment and refresh its origin."""
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.simulator.terrain_levels[:] = level
    no_change = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.simulator.update_terrain_curriculum(env_ids, no_change, no_change)
    return env_ids


def _reset_for_level(env, level, forward_command):
    env_ids = _set_global_level(env, level)
    env.reset_idx(env_ids)
    _force_commands(env, forward_command)
    env.compute_observations()
    return _unpack_observations(env.get_observations())


class TerminalStateCapture:
    """Evaluation-only reset hook that snapshots terminal state before auto-reset."""

    def __init__(self, env):
        self.env = env
        self.enabled = False
        self.states = {}
        self.original_reset_idx = env.reset_idx

        def captured_reset_idx(env_ids):
            if self.enabled:
                for env_id in env_ids.detach().cpu().tolist():
                    self.states[env_id] = {
                        "base_pos": env.simulator.base_pos[env_id].detach().clone(),
                        "base_lin_vel": env.simulator.base_lin_vel[env_id].detach().clone(),
                        "base_ang_vel_z": env.simulator.base_ang_vel[env_id, 2].detach().clone(),
                        "timeout": bool(env.time_out_buf[env_id].item()),
                    }
            return self.original_reset_idx(env_ids)

        env.reset_idx = captured_reset_idx

    def begin_step(self):
        self.states.clear()
        self.enabled = True

    def end_step(self):
        self.enabled = False


class EpisodeStats:
    """Accumulate episode-level rows and per-difficulty/overall summaries."""

    def __init__(self, num_envs, device, metadata, success_distance):
        self.metadata = metadata
        self.success_distance = success_distance
        self.rows = []
        self.start_x = torch.zeros(num_envs, device=device)
        self.max_progress = torch.zeros(num_envs, device=device)
        self.steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.lin_error_sum = torch.zeros(num_envs, device=device)
        self.ang_error_sum = torch.zeros(num_envs, device=device)

    def start_all(self, base_pos):
        self.start_x.copy_(base_pos[:, 0])
        self.max_progress.zero_()
        self.steps.zero_()
        self.lin_error_sum.zero_()
        self.ang_error_sum.zero_()

    def clear_ids(self, env_ids, base_pos):
        if env_ids.numel() == 0:
            return
        self.start_x[env_ids] = base_pos[env_ids, 0]
        self.max_progress[env_ids] = 0.0
        self.steps[env_ids] = 0
        self.lin_error_sum[env_ids] = 0.0
        self.ang_error_sum[env_ids] = 0.0

    def add_step(self, base_pos, lin_vel, ang_vel_z, command, terminal_states):
        self.steps += 1
        step_pos = base_pos[:, 0].clone()
        step_lin = lin_vel[:, :2].clone()
        step_ang = ang_vel_z.clone()
        for env_id, state in terminal_states.items():
            step_pos[env_id] = state["base_pos"][0]
            step_lin[env_id] = state["base_lin_vel"][:2]
            step_ang[env_id] = state["base_ang_vel_z"]
        progress = torch.clamp(step_pos - self.start_x, min=0.0, max=self.success_distance)
        self.max_progress = torch.maximum(self.max_progress, progress)
        target_xy = torch.zeros_like(step_lin)
        target_xy[:, 0] = command
        self.lin_error_sum += torch.linalg.vector_norm(target_xy - step_lin, dim=1)
        self.ang_error_sum += torch.abs(step_ang)

    def record(self, env_id, level, difficulty_value, success, reason):
        steps = int(self.steps[env_id].item())
        denominator = max(steps, 1)
        row = dict(self.metadata)
        row.update({
            "episode_index": len(self.rows),
            "env_id": int(env_id),
            "difficulty_level": int(level),
            "difficulty_value": float(difficulty_value),
            "success": bool(success),
            "max_forward_progress_m": float(self.max_progress[env_id].item()),
            "episode_steps": steps,
            "termination_reason": reason,
            "mean_linear_tracking_error": float(self.lin_error_sum[env_id].item() / denominator),
            "mean_angular_tracking_error": float(self.ang_error_sum[env_id].item() / denominator),
        })
        self.rows.append(row)

    @staticmethod
    def _summarize(rows, metadata):
        def mean(key):
            return float(np.mean([row[key] for row in rows])) if rows else float("nan")

        result = dict(metadata)
        result.update({
            "num_episodes": len(rows),
            "success_rate_percent": 100.0 * mean("success"),
            "average_distance_m": mean("max_forward_progress_m"),
            "mean_linear_tracking_error": mean("mean_linear_tracking_error"),
            "mean_angular_tracking_error": mean("mean_angular_tracking_error"),
        })
        return result

    def payload(self, difficulty_values, levels_requested, episodes_per_level, steps_run):
        per_difficulty = []
        for level, value in enumerate(difficulty_values):
            rows = [row for row in self.rows if row["difficulty_level"] == level]
            summary = self._summarize(rows, self.metadata)
            summary.update({"difficulty_level": level, "difficulty_value": float(value)})
            per_difficulty.append(summary)
        overall = self._summarize(self.rows, self.metadata)
        overall.update({
            "difficulty_levels_requested": levels_requested,
            "episodes_per_level_requested": episodes_per_level,
            "simulation_steps": steps_run,
            "complete": len(self.rows) == levels_requested * episodes_per_level,
        })
        return {
            "overall": overall,
            "per_difficulty": per_difficulty,
            "episodes": self.rows,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }


def _difficulty_values(args):
    if args.skill in DIFFICULTY_RANGES:
        start, end, _ = DIFFICULTY_RANGES[args.skill]
        return np.linspace(start, end, args.num_difficulty_levels).tolist()
    # Rough keeps its existing curriculum; report the generator's row fraction.
    return [level / args.num_difficulty_levels for level in range(args.num_difficulty_levels)]


def evaluate(env, policy, args, metadata):
    difficulty_values = _difficulty_values(args)
    terminal_capture = TerminalStateCapture(env)
    obs_buf, obs_history, depth = _reset_for_level(env, 0, args.forward_command)
    stats = EpisodeStats(env.num_envs, env.device, metadata, args.success_distance)
    stats.start_all(env.simulator.base_pos)
    completed_at_level = 0
    level = 0
    steps_run = 0

    while level < args.num_difficulty_levels and steps_run < args.num_steps:
        _force_commands(env, args.forward_command)
        with torch.no_grad():
            actions = policy(obs_buf.detach(), obs_history.detach(), depth.detach())
        terminal_capture.begin_step()
        try:
            step_output = env.step(actions.detach())
        finally:
            terminal_capture.end_step()
        obs_buf, _, obs_history, _, _, _, dones, _, depth = step_output
        steps_run += 1

        stats.add_step(
            env.simulator.base_pos,
            env.simulator.base_lin_vel,
            env.simulator.base_ang_vel[:, 2],
            args.forward_command,
            terminal_capture.states,
        )
        done_ids = set(dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist())
        success_ids = set(
            (stats.max_progress >= args.success_distance)
            .nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        )
        completed_ids = sorted(done_ids | success_ids)
        remaining = args.episodes_per_level - completed_at_level
        selected_ids = completed_ids[:remaining]

        for env_id in selected_ids:
            success = env_id in success_ids
            if success:
                reason = "success_distance"
            else:
                terminal = terminal_capture.states.get(env_id, {})
                reason = "timeout" if terminal.get("timeout", False) else "termination"
            stats.record(env_id, level, difficulty_values[level], success, reason)
        completed_at_level += len(selected_ids)

        if completed_at_level == args.episodes_per_level:
            level += 1
            completed_at_level = 0
            if level < args.num_difficulty_levels:
                obs_buf, obs_history, depth = _reset_for_level(
                    env, level, args.forward_command
                )
                stats.start_all(env.simulator.base_pos)
            else:
                # Leave no completed or partial episode running after the final quota.
                env.reset_idx(torch.arange(env.num_envs, device=env.device))
            continue

        # Natural dones have reset already; reset successful non-done envs preemptively.
        preempt_ids = sorted(success_ids - done_ids)
        if preempt_ids:
            ids = torch.tensor(preempt_ids, device=env.device, dtype=torch.long)
            env.reset_idx(ids)
            _force_commands(env, args.forward_command)
            env.compute_observations()
            obs_buf, obs_history, depth = _unpack_observations(env.get_observations())
        reset_ids = sorted(done_ids | set(preempt_ids))
        if reset_ids:
            ids = torch.tensor(reset_ids, device=env.device, dtype=torch.long)
            stats.clear_ids(ids, env.simulator.base_pos)

    return stats.payload(
        difficulty_values, args.num_difficulty_levels, args.episodes_per_level, steps_run
    )


def save_results(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.lower().endswith(".csv"):
        rows = payload["episodes"]
        with open(path, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        json_path = os.path.splitext(path)[0] + "_summary.json"
    else:
        json_path = path if path.lower().endswith(".json") else path + ".json"
    with open(json_path, "w") as output:
        json.dump(payload, output, indent=2, allow_nan=True)
    print(f"Saved evaluation results to: {json_path}")


def evaluate_policy(args):
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args.seed
    override_configs(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    if args.load_run:
        train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.ckpt
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    metadata = {
        "skill": args.skill,
        "terrain": SKILL_TO_TERRAIN[args.skill],
        "seed": args.seed,
        "task": args.task,
        "load_run": args.load_run if args.load_run is not None else "latest",
        "checkpoint": args.ckpt if args.ckpt >= 0 else "latest",
        "forward_command_mps": args.forward_command,
        "success_distance_m": args.success_distance,
        "disabled_randomizations": sorted(
            name for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items()
            if not enabled
        ),
    }
    payload = evaluate(env, policy, args, metadata)
    overall = payload["overall"]
    print("=== Evaluation Results ===")
    print(f"Episodes completed:     {overall['num_episodes']}")
    print(f"Success rate:           {overall['success_rate_percent']:.2f}%")
    print(f"Average distance:       {overall['average_distance_m']:.3f} m")
    print(f"Mean lin. vel. error:   {overall['mean_linear_tracking_error']:.4f} m/s")
    print(f"Mean ang. vel. error:   {overall['mean_angular_tracking_error']:.4f} rad/s")
    if not overall["complete"]:
        print(f"WARNING: num_steps safety cap reached after {overall['simulation_steps']} steps")
    if args.save_path:
        save_results(payload, args.save_path)
    return payload


if __name__ == "__main__":
    evaluate_policy(get_args())
