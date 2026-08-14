"""
eval_classifier_multiterrain.py

High-level evaluation harness for a JIT-swap policy driven by an online
terrain classifier, run on a `multiterrain` environment.

It reports three headline metrics:
  1. Success rate      - fraction of episodes that end via time-out
                          (i.e. the robot did NOT fall / get terminated early).
  2. Distance covered   - per-episode straight-line displacement (start -> end),
                          reported as mean/median/std across completed episodes.
  3. Classification accy - fraction of classifier predictions that match the
                          *ground-truth* terrain patch the robot is currently
                          standing/looking at (derived from env terrain labels),
                          plus a per-class confusion breakdown.

Assumptions (matches the `depth_waq`-style branch in play.py):
  - The task exposes a depth camera: env.depth_sensor_output (B, 1, H, W) or similar.
  - `env.get_observations()` / `env.step()` follow the depth_waq signature:
        obs_buf, priv_obs, obs_history, explicit_labels, next_states, depth = env.get_observations()
        obs_buf, priv_obs, obs_history, explicit_labels, next_states, rews, dones, infos, depth = env.step(actions)
  - The JIT policy exposes `.swap(idx)` to switch between loaded sub-policies
    ("loras"), the same convention used by play.py's --jit flag.
  - Ground-truth terrain per patch is available at env.simulator._terrain.labels
    (row, col) -> int label, decodable via legged_gym.utils.terrain_vars.TERRAIN_KEYS.

If your task/env differs, adjust `get_observations_depth_waq` / `step_depth_waq`
and `get_ground_truth_label` accordingly - they're isolated on purpose.

Usage:
    python eval_classifier_multiterrain.py \
        --task go2_depth_waq \
        --jit /path/to/swap_policy.jit.pt \
        --terrain_detector_jit /path/to/classifier.jit.pt \
        --num_envs 200 --num_episodes 20 --headless

    # Bayesian-filtered classifier instead of a raw jit classifier:
    python eval_classifier_multiterrain.py \
        --task go2_depth_waq \
        --jit /path/to/swap_policy.jit.pt \
        --baysian_filter /path/to/bayes_ckpt_dir --num_envs 200 --headless
"""

import os
import json
import argparse
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.terrain_vars import TERRAIN_INDEX, TERRAIN_KEYS

try:
    from legged_gym.scripts.play import configure_runtime_device, init_genesis
except Exception:
    configure_runtime_device = None
    init_genesis = None


# --------------------------------------------------------------------------- #
# Terrain -> lora index map, mirrors the swap logic used in play.py
# --------------------------------------------------------------------------- #
def label_to_lora(label: str) -> int:
    if label == "random_uniform":
        return -1
    if "stairs" in label:
        return 1
    if label == "gap":
        return 0
    if label in ("pit", "center_platform"):
        return 2
    return -1


# Every distinct lora id label_to_lora can produce - one JIT policy copy gets
# loaded and pre-swapped to each of these so per-env dispatch is a pure lookup.
LORA_IDS = sorted({-1, 0, 1, 2})


class PerTerrainPolicySet:
    """Holds one independently-swapped JIT policy copy per terrain lora id, so
    each env's action is computed by the sub-policy matching *that env's own*
    classifier prediction, instead of swapping a single shared policy off a
    global (e.g. majority-vote) decision.
    """

    def __init__(self, jit_path, device, lora_ids=LORA_IDS):
        self.device = device
        self.policies = {}
        for lora_id in lora_ids:
            p = torch.jit.load(jit_path, map_location=device)
            p.swap(lora_id)
            self.policies[lora_id] = p

    def act(self, obs_buf, obs_history, depth, assigned_lora):
        """assigned_lora: (num_envs,) LongTensor of per-env lora ids.

        Runs each loaded sub-policy only on the subset of envs assigned to it,
        and scatters the results back into a single (num_envs, action_dim)
        tensor.
        """
        assigned_lora = assigned_lora.to(obs_buf.device)
        actions = None
        for lora_id, p in self.policies.items():
            idx = (assigned_lora == lora_id).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            sub_actions = p(obs_buf[idx].detach(), obs_history[idx].detach(), depth[idx].detach())
            if actions is None:
                actions = torch.zeros(
                    obs_buf.shape[0], sub_actions.shape[-1],
                    device=sub_actions.device, dtype=sub_actions.dtype,
                )
            actions[idx] = sub_actions
        return actions


# --------------------------------------------------------------------------- #
# Ground-truth terrain lookup (reused from play.py's get_viewed_terrain_idx)
# --------------------------------------------------------------------------- #
def get_viewed_terrain_idx(env, look_ahead_frac: float = 0.75):
    far_plane = 4.0
    look_ahead_dist = far_plane * look_ahead_frac

    base_pos = env.simulator.base_pos
    base_pos_xy = base_pos[..., :2]
    heading = env.heading

    single = base_pos.dim() == 1
    if single:
        base_pos = base_pos.unsqueeze(0)
        heading = heading.unsqueeze(0) if torch.is_tensor(heading) else torch.tensor([heading])

    device = base_pos.device
    heading = heading.to(device)

    look_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    look_point = base_pos[:, :2] + look_dir * look_ahead_dist
    query_point = torch.where((base_pos[:, 2] < 0.25).unsqueeze(-1), base_pos_xy, look_point)

    origins = env.simulator._terrain_origins.to(device)
    num_rows, num_cols = origins.shape[0], origins.shape[1]
    origins_xy = origins[..., :2].reshape(-1, 2)

    dists = torch.cdist(query_point, origins_xy)
    idx = torch.argmin(dists, dim=-1)
    row_col = torch.stack([idx // num_cols, idx % num_cols], dim=-1)

    if single:
        idx = idx.squeeze(0)
        row_col = row_col.squeeze(0)
    return idx, row_col


def get_ground_truth_labels(env):
    """Returns a (num_envs,) LongTensor of ground-truth terrain label ids."""
    _, row_col = get_viewed_terrain_idx(env)
    row_col = row_col.cpu()
    return env.simulator._terrain.labels[row_col[:, 0], row_col[:, 1]]


# --------------------------------------------------------------------------- #
# Classifier loading
# --------------------------------------------------------------------------- #

def build_classifier(args):
    from legged_gym.scripts.depth_data_pipeline.train_feature_nn import TerrainDepthFeatureClassifierNN
    from legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn import TerrainDepthClassifierNN
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import *
    from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import SobelDepthTerrainFeatureExtractor
    from pathlib import Path 
    all_data_files = Path(args.baysian_filter)
    classifier_file = all_data_files / "classifier.pt"
    bayes_filter_file = all_data_files / "bayes_filter.pt"
    assert classifier_file.is_file() and bayes_filter_file.is_file()
    extractor_file = all_data_files / "extractor.pt"
    nn_model_args_file = all_data_files / "nn_model_args.pt"

    if nn_model_args_file.is_file():
        nn_model_args = torch.load(nn_model_args_file)
        class_name = nn_model_args.pop("cls")
        model = eval(class_name)(**nn_model_args)
        classifier = NeuralClassifierAdapter.load(classifier_file, model)
        extractor = lambda x, y, z : x
    else:
        classifier = PCAWhitenedRBFPrototypeClassifier.load(classifier_file)
        extractor_base = SobelDepthTerrainFeatureExtractor.load(extractor_file)
        extractor = lambda x, y, z : extractor_base.extract_batch(x, y, z)

    bayesian_terrain_filter = BayesianTerrainFilter.load(bayes_filter_file)

    def predict_fn(_depth, _euler, _angve):
        inputs = extractor(_depth, _euler, _angve)
        classifier_probabilities, _ = classifier.predict_class_distribution(inputs)
        predicted, _, _ = run_filter_sequences(bayesian_terrain_filter, classifier_probabilities)
        return predicted

    def reset_fn():
        bayesian_terrain_filter.reset()
    
    return predict_fn, reset_fn

# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #
def get_args():
    parser = argparse.ArgumentParser(description="Eval JIT-swap policy + terrain classifier on multiterrain")
    parser.add_argument("--task", type=str, default="go2", help="task name (should be a depth_waq-style task)")
    parser.add_argument("--gpu", type=str, default="cuda:0")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--num_episodes", type=int, default=20,
                         help="approx. number of episode lengths to run for (per env)")

    parser.add_argument("--jit", type=str, required=True, help="path to jit-scripted swap policy")
    parser.add_argument("--baysian_filter", type=str, default="", help="path to a bayesian-filtered classifier checkpoint dir")
    parser.add_argument("--classify_every", type=int, default=5,
                         help="run classifier + swap every N sim steps (matches depth cam rate)")
    parser.add_argument("--out_dir", type=str, default=None, help="where to write results (json/csv)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.baysian_filter:
        parser.error("Must supply --baysian_filter")

    if args.cpu:
        args.gpu = "cpu"
    return args


# --------------------------------------------------------------------------- #
# Env config overrides: force a multiterrain layout
# --------------------------------------------------------------------------- #
def override_configs_multiterrain(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs
    env_cfg.asset.terminate_after_contacts_on = []
    env_cfg.init_state.yaw_random_scale = 0
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = env_cfg.env.num_envs

    env_cfg.commands.custom_command_curriculum = False
    env_cfg.viewer.rendered_envs_idx = list(range(min(env_cfg.env.num_envs, args.num_envs)))
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1

    # update this as needed
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.border_size = 5.0
    env_cfg.terrain.platform_size = 3.0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.custom_selected = True

    terrain_types = [
        {"type": "terrain_utils.random_uniform_terrain", "min_height": -0.05,
         "max_height": 0.05, "step": 0.005, "downsampled_scale": 0.2},
        {"type": "terrain_utils.gap_terrain", "gap_size": 0.5, "platform_size": 3.0},
        {"type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
         "step_height": -0.2, "platform_size": 3.0},
        {"type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
         "step_height": 0.2, "platform_size": 3.0},
        {"type": "terrain_utils.pit_terrain", "depth": 0.2, "platform_size": 3.0},
    ]
    rng = random.Random(args.seed)
    env_cfg.terrain.terrain_map = [
        rng.choice(terrain_types).copy()
        for _ in range(env_cfg.terrain.num_rows * env_cfg.terrain.num_cols)
    ]

    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.asset.fix_base_link = False


# --------------------------------------------------------------------------- #
# Episode bookkeeping
# --------------------------------------------------------------------------- #
class EpisodeStats:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.start_pos = None
        self.successes = 0
        self.failures = 0
        self.completed_distance = []

    def on_step_start_positions(self, base_pos):
        if self.start_pos is None:
            self.start_pos = base_pos[:, :2].clone()

    def on_done(self, base_pos, dones, infos):
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return

        distance = torch.norm(base_pos[done_ids, :2] - self.start_pos[done_ids], dim=-1)
        self.completed_distance.extend(distance.cpu().tolist())

        time_out = infos.get("time_out", None)
        if time_out is not None:
            time_out = time_out.to(self.device)
            succ = time_out[done_ids].sum().item()
            fail = done_ids.numel() - succ
        else:
            # Fallback: no time_out signal available -> treat all resets as failures
            # unless caller marks otherwise.
            succ = 0
            fail = done_ids.numel()

        self.successes += int(succ)
        self.failures += int(fail)

        # reset start positions for envs that just finished an episode
        self.start_pos[done_ids] = base_pos[done_ids, :2].clone()

    def summary(self):
        total = self.successes + self.failures
        dist = np.array(self.completed_distance) if self.completed_distance else np.array([0.0])
        return {
            "num_completed_episodes": total,
            "success_rate": self.successes / total if total > 0 else float("nan"),
            "distance_mean": float(dist.mean()),
            "distance_median": float(np.median(dist)),
            "distance_std": float(dist.std()),
        }


class ClassificationStats:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.confusion = defaultdict(lambda: defaultdict(int))  # gt -> pred -> count

    def update(self, gt_labels, pred_labels):
        for gt, pred in zip(gt_labels, pred_labels):
            self.total += 1
            self.confusion[gt][pred] += 1
            if gt == pred:
                self.correct += 1

    def summary(self):
        per_class = {}
        for gt, preds in self.confusion.items():
            n = sum(preds.values())
            per_class[gt] = {
                "support": n,
                "accuracy": preds.get(gt, 0) / n if n > 0 else float("nan"),
            }
        return {
            "overall_accuracy": self.correct / self.total if self.total > 0 else float("nan"),
            "total_predictions": self.total,
            "per_class": per_class,
        }


# --------------------------------------------------------------------------- #
# Main eval loop
# --------------------------------------------------------------------------- #
def run_eval(args):
    if configure_runtime_device is not None:
        configure_runtime_device(args)
    if init_genesis is not None and "genesis" in globals().get("SIMULATOR", ""):
        init_genesis(args, gs)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    override_configs_multiterrain(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    policy_set = PerTerrainPolicySet(args.jit, device=args.gpu if not args.cpu else "cpu")

    predict_fn, reset_fn = build_classifier(args)

    ep_stats = EpisodeStats(env.num_envs, env.device)
    cls_stats = ClassificationStats()

    obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, depth = env.get_observations()
    ep_stats.on_step_start_positions(env.simulator.base_pos)

    # Per-env lora assignment, updated at each classification tick and held
    # fixed for the steps in between. Starts on the baseline/random_uniform
    # policy for every env until the first classification pass runs.
    assigned_lora = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    total_steps = int(args.num_episodes * env.max_episode_length)
    for i in range(total_steps):
        actions = policy_set.act(obs_buf.detach(), obs_history.detach(), depth.detach(), assigned_lora.detach())
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos, depth = \
            env.step(actions.detach())

        if i % args.classify_every == 0:
            gt_labels_id = get_ground_truth_labels(env)
            gt_labels = [TERRAIN_KEYS[int(l)].lower() for l in gt_labels_id]
            pred_labels = []
            _depth = env.depth_sensor_output.squeeze().detach().cpu().clone()
            _euler = env.simulator._base_euler.detach().cpu().clone()
            _angve = env.simulator.base_ang_vel.detach().cpu().clone()
            pred_labels.append(predict_fn(_depth, _euler, _angve))
            cls_stats.update(gt_labels, pred_labels)
            # Each env is routed to the sub-policy matching *its own*
            # classifier prediction - no global/majority swap.
            assigned_lora = torch.tensor(
                [label_to_lora(l) for l in gt_labels], dtype=torch.long, device=env.device
            )

        ep_stats.on_done(env.simulator.base_pos, dones, infos)

        if dones[0]:
            reset_fn()

    return ep_stats.summary(), cls_stats.summary()


def main():
    args = get_args()
    ep_summary, cls_summary = run_eval(args)

    print("\n=== Success / Distance ===")
    print(f"  completed episodes : {ep_summary['num_completed_episodes']}")
    print(f"  success rate        : {ep_summary['success_rate']:.3f}")
    print(f"  distance (m) mean   : {ep_summary['distance_mean']:.3f}")
    print(f"  distance (m) median : {ep_summary['distance_median']:.3f}")
    print(f"  distance (m) std    : {ep_summary['distance_std']:.3f}")

    print("\n=== Classifier accuracy ===")
    print(f"  overall accuracy : {cls_summary['overall_accuracy']:.3f} "
          f"({cls_summary['total_predictions']} predictions)")
    for gt, stats in cls_summary["per_class"].items():
        print(f"    {gt:>20s} : acc={stats['accuracy']:.3f}  (n={stats['support']})")

    out_dir = args.out_dir or os.path.join(LEGGED_GYM_ROOT_DIR, "exp_logs", "classifier_eval")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"eval_{args.task}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "success_distance": ep_summary, "classification": cls_summary}, f, indent=2)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()