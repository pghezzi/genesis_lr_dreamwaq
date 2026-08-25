"""Evaluate terrain classification, skill selection, and multi-terrain navigation."""
from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.terrain_vars import TERRAIN_KEYS

import argparse
import copy
import csv
import json
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch



CANONICAL_CLASSES = ["rough", "gap", "pit", "stairs"]
SKILL_SPEEDS = {"rough": 0.8, "gap": 1.5, "pit": 1.2, "stairs": 1.2}
SKILL_LORA = {"rough": -1, "gap": 0, "stairs": 1, "pit": 2}
METHOD_TO_APPROACH = {
    "RBF Prototype": "rbf_prototype",
    "RBF SVM": "rbf_svm",
    "feature NN": "feature_nn",
    "raw-depth NN": "raw_depth_nn",
}

# Keep evaluation randomization identical to low_level_evaluation.py.
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


def configure_runtime_device(args):
    """Normalize the evaluator device while retaining physical-GPU masking."""
    if args.cpu:
        args.gpu = "cpu"
        args.device = "cpu"
        return args
    requested = str(args.gpu).lower()
    if requested.isdigit():
        requested = f"cuda:{requested}"
    if requested == "cuda":
        requested = "cuda:0"
    if not requested.startswith("cuda:") or not requested.split(":", 1)[1].isdigit():
        raise ValueError("--gpu must be cuda, cuda:N, or a numeric GPU index")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        visible_ids = [value.strip() for value in visible.split(",") if value.strip()]
        index = int(requested.split(":", 1)[1])
        if index < len(visible_ids):
            requested = f"cuda:{index}"
        elif str(index) in visible_ids:
            requested = f"cuda:{visible_ids.index(str(index))}"
        else:
            raise ValueError(f"GPU {index} is unavailable under CUDA_VISIBLE_DEVICES={visible}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested.split(":", 1)[1]
        requested = "cuda:0"
    args.gpu = requested
    args.device = requested
    return args


def init_genesis(args, gs):
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu:
        torch.cuda.set_device(torch.device(args.gpu))


def canonicalize_label(label):
    if torch.is_tensor(label):
        label = label.item()
    if isinstance(label, (int, np.integer)):
        if not 0 <= int(label) < len(TERRAIN_KEYS):
            raise ValueError(f"Unknown terrain label id {label}")
        label = TERRAIN_KEYS[int(label)]
    value = str(label).lower()
    if value in ("random_uniform", "pyramid_sloped", "rough", "baseline"):
        return "rough"
    if value in ("stairs", "upwards_stairs"):
        return "stairs"
    if value in ("pit", "center_platform", "climb"):
        return "pit"
    if value in ("gap", "leap"):
        return "gap"
    raise ValueError(f"Cannot map terrain label {label!r} to a skill class")


def label_to_lora(label):
    return SKILL_LORA[canonicalize_label(label)]


class PerTerrainPolicySet:
    """Preload one independently swapped policy per skill for batched dispatch."""

    def __init__(self, jit_path, device):
        self.policies = {}
        for lora_id in sorted(set(SKILL_LORA.values())):
            policy = torch.jit.load(jit_path, map_location=device)
            policy.swap(lora_id)
            self.policies[lora_id] = policy

    def act(self, obs_buf, obs_history, depth, assigned_lora):
        actions = None
        for lora_id, policy in self.policies.items():
            indices = (assigned_lora == lora_id).nonzero(as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            sub_actions = policy(
                obs_buf[indices].detach(), obs_history[indices].detach(), depth[indices].detach()
            )
            if actions is None:
                actions = torch.zeros(
                    obs_buf.shape[0], sub_actions.shape[-1],
                    device=sub_actions.device, dtype=sub_actions.dtype,
                )
            actions[indices] = sub_actions
        if actions is None:
            raise RuntimeError("No environments were assigned to a known skill policy")
        return actions


def get_viewed_terrain_idx(env, look_ahead_frac=0.75):
    """Return the existing look-ahead terrain-cell lookup used by the oracle."""
    look_ahead_dist = 4.0 * look_ahead_frac
    base_pos = env.simulator.base_pos
    heading = env.heading.to(base_pos.device)
    look_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    look_point = base_pos[:, :2] + look_dir * look_ahead_dist
    query_point = torch.where(
        (base_pos[:, 2] < 0.25).unsqueeze(-1), base_pos[:, :2], look_point
    )
    origins = env.simulator._terrain_origins.to(base_pos.device)
    num_cols = origins.shape[1]
    distances = torch.cdist(query_point, origins[..., :2].reshape(-1, 2))
    indices = torch.argmin(distances, dim=-1)
    row_col = torch.stack([indices // num_cols, indices % num_cols], dim=-1)
    return indices, row_col


def get_ground_truth_labels(env, look_ahead_frac=0.75):
    """Decode raw labels solely from the environment terrain-label grid."""
    _, row_col = get_viewed_terrain_idx(env, look_ahead_frac)
    row_col = row_col.detach().cpu()
    label_ids = env.simulator._terrain.labels[row_col[:, 0], row_col[:, 1]]
    raw = [TERRAIN_KEYS[int(label_id)] for label_id in label_ids]
    return raw, [canonicalize_label(label) for label in raw]


class RuntimeClassifier:
    def __init__(self, classifier, approach, extractor=None, standardizer=None):
        self.classifier = classifier
        self.approach = approach
        self.extractor = extractor
        self.standardizer = standardizer
        self.class_ids = list(classifier.class_ids)

    def predict(self, depth, euler, angular_velocity):
        if self.approach == "raw_depth_nn":
            inputs = depth
        else:
            inputs = self.extractor.extract_batch(depth, euler, angular_velocity)
            if self.standardizer is not None:
                inputs = self.standardizer.transform(inputs)
        probabilities, class_ids = self.classifier.predict_class_distribution(inputs)
        if list(class_ids) != self.class_ids:
            raise RuntimeError("Classifier returned a class ordering different from classifier.class_ids")
        if probabilities.ndim != 2 or probabilities.shape[1] != len(self.class_ids):
            raise RuntimeError(
                f"Classifier must return [B,{len(self.class_ids)}], got {tuple(probabilities.shape)}"
            )
        return probabilities.detach()


def _classifier_artifacts(classifier_dir):
    root = Path(classifier_dir)
    classifier_file = root / "classifier.pt"
    if not classifier_file.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {classifier_file}")
    return root, classifier_file


def _feature_parts(root, device):
    from legged_gym.utils.depth_terrain_classifier.depth_terrain_classifier import (
        SobelDepthTerrainFeatureExtractor,
    )
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        FeatureStandardizer,
    )

    extractor_file = root / "extractor.pt"
    if not extractor_file.is_file():
        raise FileNotFoundError(f"Missing engineered-feature extractor: {extractor_file}")
    extractor = SobelDepthTerrainFeatureExtractor.load(extractor_file, device=device)
    standardizer_file = root / "standardizer.pt"
    standardizer = (
        FeatureStandardizer.load(standardizer_file) if standardizer_file.is_file() else None
    )
    return extractor, standardizer


def load_rbf_prototype(classifier_dir, device):
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        PCAWhitenedRBFPrototypeClassifier,
    )
    root, checkpoint = _classifier_artifacts(classifier_dir)
    extractor, standardizer = _feature_parts(root, device)
    classifier = PCAWhitenedRBFPrototypeClassifier.load(checkpoint, map_location=device)
    return RuntimeClassifier(classifier, "rbf_prototype", extractor, standardizer)


def load_rbf_svm(classifier_dir, device):
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import RBFSVM
    root, checkpoint = _classifier_artifacts(classifier_dir)
    extractor, standardizer = _feature_parts(root, device)
    classifier = RBFSVM.load(checkpoint, map_location=device)
    return RuntimeClassifier(classifier, "rbf_svm", extractor, standardizer)


def _load_neural(classifier_dir, device, raw_depth):
    from legged_gym.scripts.depth_data_pipeline.train_feature_nn import TerrainDepthFeatureClassifierNN
    from legged_gym.scripts.depth_data_pipeline.train_raw_depth_nn import TerrainDepthClassifierNN
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        NeuralClassifierAdapter,
    )
    root, checkpoint = _classifier_artifacts(classifier_dir)
    args_file = root / "nn_model_args.pt"
    if not args_file.is_file():
        raise FileNotFoundError(f"Missing neural model arguments: {args_file}")
    model_args = dict(torch.load(args_file, map_location="cpu", weights_only=False))
    class_name = model_args.pop("cls")
    model_types = {
        "TerrainDepthFeatureClassifierNN": TerrainDepthFeatureClassifierNN,
        "TerrainDepthClassifierNN": TerrainDepthClassifierNN,
    }
    expected = "TerrainDepthClassifierNN" if raw_depth else "TerrainDepthFeatureClassifierNN"
    if class_name != expected:
        raise ValueError(f"Expected {expected} model arguments, found {class_name}")
    model = model_types[class_name](**model_args)
    classifier = NeuralClassifierAdapter.load(checkpoint, model, device=device)
    if raw_depth:
        return RuntimeClassifier(classifier, "raw_depth_nn")
    extractor, standardizer = _feature_parts(root, device)
    return RuntimeClassifier(classifier, "feature_nn", extractor, standardizer)


def load_feature_nn(classifier_dir, device):
    return _load_neural(classifier_dir, device, raw_depth=False)


def load_raw_depth_nn(classifier_dir, device):
    return _load_neural(classifier_dir, device, raw_depth=True)


CLASSIFIER_LOADERS = {
    "rbf_prototype": load_rbf_prototype,
    "rbf_svm": load_rbf_svm,
    "feature_nn": load_feature_nn,
    "raw_depth_nn": load_raw_depth_nn,
}


def resolve_classifier_approach(args):
    if args.classifier_approach != "auto":
        return args.classifier_approach
    root = Path(args.classifier_dir)
    results_file = root / "results.json"
    if results_file.is_file():
        with results_file.open(encoding="utf-8") as stream:
            method = json.load(stream).get("method")
        if method in METHOD_TO_APPROACH:
            return METHOD_TO_APPROACH[method]
    model_args_file = root / "nn_model_args.pt"
    if model_args_file.is_file():
        model_args = torch.load(model_args_file, map_location="cpu", weights_only=False)
        neural_types = {
            "TerrainDepthFeatureClassifierNN": "feature_nn",
            "TerrainDepthClassifierNN": "raw_depth_nn",
        }
        if model_args.get("cls") in neural_types:
            return neural_types[model_args["cls"]]
    checkpoint_file = root / "classifier.pt"
    if checkpoint_file.is_file():
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if "kernel_basis" in state:
            return "rbf_svm"
        if "prototypes" in state and "pca_components" in state:
            return "rbf_prototype"
    raise ValueError("Could not infer classifier approach from results.json or saved artifacts")


def resolve_classifier_approach_auto(classifier_dir):
    root = Path(classifier_dir)
    results_file = root / "results.json"
    if results_file.is_file():
        with results_file.open(encoding="utf-8") as stream:
            method = json.load(stream).get("method")
        if method in METHOD_TO_APPROACH:
            return METHOD_TO_APPROACH[method]
    model_args_file = root / "nn_model_args.pt"
    if model_args_file.is_file():
        model_args = torch.load(model_args_file, map_location="cpu", weights_only=False)
        neural_types = {
            "TerrainDepthFeatureClassifierNN": "feature_nn",
            "TerrainDepthClassifierNN": "raw_depth_nn",
        }
        if model_args.get("cls") in neural_types:
            return neural_types[model_args["cls"]]
    checkpoint_file = root / "classifier.pt"
    if checkpoint_file.is_file():
        state = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        if "kernel_basis" in state:
            return "rbf_svm"
        if "prototypes" in state and "pca_components" in state:
            return "rbf_prototype"
    raise ValueError("Could not infer classifier approach from results.json or saved artifacts")

def create_classifier(classifier_dir, device):
    approach = resolve_classifier_approach_auto(classifier_dir)
    return CLASSIFIER_LOADERS[approach](classifier_dir, device)
    
def auto_load_checkpoint_bayes_filter(classifier_dir, device):
    path = Path(classifier_dir) / "bayes_filter.pt"
    return _load_bayes_template(path, device)


def load_paired_bayes_filter(args, device):
    path = Path(args.classifier_dir) / "bayes_filter.pt"
    return _load_bayes_template(path, device), path


def load_checkpoint_bayes_filter(args, device):
    if not args.bayes_filter_path:
        raise ValueError("--bayes_filter_path is required for checkpoint filters")
    path = Path(args.bayes_filter_path)
    return _load_bayes_template(path, device), path


def _load_bayes_template(path, device):
    from legged_gym.utils.depth_terrain_classifier.terrain_classifier_bayes_streaming_prototype_rbf import (
        BayesianTerrainFilter,
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Bayes filter checkpoint: {path}")
    result = BayesianTerrainFilter.load(path, device=device)
    result.reset()
    return result


BAYES_FILTER_LOADERS = {
    "paired": load_paired_bayes_filter,
    "checkpoint": load_checkpoint_bayes_filter,
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate JIT skill selection and terrain classification on fixed tracks"
    )
    parser.add_argument("--task", default="go2_depth_waq_lora")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=10)
    parser.add_argument("--episodes_per_track", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=100000, help="safety cap only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--finish_margin", type=float, default=0.25)
    parser.add_argument("--look_ahead_frac", type=float, default=0)
    parser.add_argument("--classify_every", type=int, default=5)
    parser.add_argument(
        "--selector_mode", choices=("oracle", "instantaneous", "bayes", "baseline"),
        default="bayes",
    )
    parser.add_argument(
        "--classifier_approach", choices=("auto",) + tuple(CLASSIFIER_LOADERS), default="auto"
    )
    parser.add_argument("--classifier_dir", required=True)
    parser.add_argument(
        "--bayes_filter_approach", choices=tuple(BAYES_FILTER_LOADERS), default="paired"
    )
    parser.add_argument("--bayes_filter_path", default=None)
    parser.add_argument("--jit", "--policy_jit", dest="jit", required=True)
    parser.add_argument("--fixed_forward_command", type=float, default=None)
    parser.add_argument("--out_dir", default=None)
    
    #not used args
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--motion_file",
        type=str,
        default=None,
        help="Optional motion file",
    )
    parser.add_argument(
        "--num_student",
        type=int,
        default=None,
        help="Number of student environments/agents",
    )
    args = parser.parse_args()
    if args.cpu:
        args.gpu = "cpu"
    if args.num_envs < 10 or args.num_envs % 10:
        parser.error("--num_envs must be 10 or a balanced multiple of 10")
    if args.episodes_per_track < 1 or args.num_steps < 1 or args.classify_every < 1:
        parser.error("episode quota, step cap, and classify interval must be positive")
    if not 0.0 <= args.difficulty <= 1.0:
        parser.error("--difficulty must be in [0,1]")
    if not 0.0 <= args.look_ahead_frac <= 1.0:
        parser.error("--look_ahead_frac must be in [0,1]")
    return args


def _eval_terrain_value(value, difficulty):
    return float(eval(str(value), {"__builtins__": {}}, {"np": np, "difficulty": difficulty}))


def make_track_layout(env_cfg, args):
    difficulty = args.difficulty
    gap_size = 0.30 + 0.70 * difficulty
    pit_depth = 0.25 + 0.25 * difficulty
    stair_height = 0.10 + 0.30 * difficulty
    rough_cfg = env_cfg.terrain.terrain_curriculum_difficulty["random_uniform_params"]
    rough_values = {
        key: _eval_terrain_value(value, difficulty) for key, value in rough_cfg.items()
    }
    definitions = {
        "random_uniform": {
            "type": "terrain_utils.random_uniform_terrain", **rough_values,
        },
        "gap": {
            "type": "terrain_utils.gap_terrain", "gap_size": 0.5,
            "platform_size": env_cfg.terrain.platform_size,
        },
        "pit": {
            "type": "terrain_utils.pit_terrain", "depth": 0.3,
            "platform_size": env_cfg.terrain.platform_size,
        },
        "upwards_stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
            "step_height": 0.25, "platform_size": env_cfg.terrain.platform_size,
        },
        "stairs": {
            "type": "terrain_utils.pyramid_stairs_terrain", "step_width": 0.4,
            "step_height": -0.25, "platform_size": env_cfg.terrain.platform_size,
        },
    }
    rng = random.Random(args.seed)
    columns = []
    for column in range(10):
        sequence = list(definitions)
        rng.shuffle(sequence)
        columns.append(sequence)
    terrain_map = [copy.deepcopy(definitions[columns[col][row]]) for row in range(5) for col in range(10)]
    logged = []
    for column, sequence in enumerate(columns):
        logged.append({
            "track_id": column,
            "sequence": sequence,
            "cells": [
                {"row": row, "raw_label": name,
                 "parameters": {k: v for k, v in definitions[name].items() if k != "type"}}
                for row, name in enumerate(sequence)
            ],
        })
    return terrain_map, logged


def override_configs_multiterrain(env_cfg, args):
    env_cfg.seed = args.seed
    env_cfg.env.num_envs = args.num_envs
    if hasattr(env_cfg.env, "num_camera_envs"):
        env_cfg.env.num_camera_envs = args.num_envs
    if hasattr(env_cfg.viewer, "rendered_envs_idx"):
        env_cfg.viewer.rendered_envs_idx = list(range(min(args.num_envs, 10)))
    env_cfg.init_state.yaw_random_scale = 0.0
    env_cfg.commands.curriculum = False
    if hasattr(env_cfg.commands, "custom_command_curriculum"):
        env_cfg.commands.custom_command_curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_cmd_prob = 0.0
    env_cfg.commands.resampling_time = 1.0e9
    env_cfg.commands.ranges.lin_vel_x = [0.8, 1.5]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.heading = [0.0, 0.0]
    
    env_cfg.asset.terminate_after_contacts_on = []

    for name, value in EVAL_DOMAIN_RANDOMIZATION_RANGES.items():
        setattr(env_cfg.domain_rand, name, list(value))
    # The mixed tracks include the obstacle skills, matching the low-level
    # evaluator's non-rough branch. These are inert when their flags are off.
    env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.max_push_vel_xy = 0.5
    for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items():
        setattr(env_cfg.domain_rand, name, enabled)

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.custom_selected = True
    terrain_map, args.track_layout = make_track_layout(env_cfg, args)
    env_cfg.terrain.terrain_map = terrain_map
    if not 0.0 <= args.finish_margin < env_cfg.terrain.terrain_length:
        raise ValueError("finish margin must be smaller than one sub-terrain length")


def assign_tracks_and_reset(env, track_ids):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.simulator.terrain_levels[:] = 0
    env.simulator.terrain_types[:] = track_ids
    unchanged = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.simulator.update_terrain_curriculum(env_ids, unchanged, unchanged)
    env.reset_idx(env_ids)
    env.compute_observations()
    return env_ids


def unpack_observations(value, expected):
    if not isinstance(value, (tuple, list)) or len(value) != expected:
        raise RuntimeError(
            f"Task must provide the depth-WaQ {'observation' if expected == 6 else 'step'} "
            f"signature with {expected} entries; received {type(value).__name__}"
        )
    return value


class TerminalCapture:
    def __init__(self, env):
        self.env = env
        self.enabled = False
        self.states = {}
        original = env.reset_idx

        def reset_idx(env_ids):
            if self.enabled:
                for env_id in env_ids.detach().cpu().tolist():
                    self.states[env_id] = {
                        "position": env.simulator.base_pos[env_id].detach().clone(),
                        "timeout": bool(env.time_out_buf[env_id].item()),
                        "episode_steps": int(env.episode_length_buf[env_id].item()),
                    }
            return original(env_ids)

        env.reset_idx = reset_idx

    def begin(self):
        self.states.clear()
        self.enabled = True

    def end(self):
        self.enabled = False


class ClassificationStats:
    def __init__(self):
        self.matrices = {
            "instantaneous": np.zeros((4, 4), dtype=np.int64),
            "bayes": np.zeros((4, 4), dtype=np.int64),
        }
        self.selected_correct = 0
        self.selected_total = 0
        self.switch_count = 0
        self.delays = []
        self.ticks = []

    def update(self, truth, instantaneous, bayes, selected, record):
        truth_index = CANONICAL_CLASSES.index(truth)
        self.matrices["instantaneous"][truth_index, CANONICAL_CLASSES.index(instantaneous)] += 1
        self.matrices["bayes"][truth_index, CANONICAL_CLASSES.index(bayes)] += 1
        self.selected_correct += int(selected == truth)
        self.selected_total += 1
        self.ticks.append(record)

    def summary(self):
        result = {}
        for name, matrix in self.matrices.items():
            support = matrix.sum(axis=1)
            correct = np.diag(matrix)
            result[name] = {
                "accuracy": float(correct.sum() / matrix.sum()) if matrix.sum() else float("nan"),
                "per_class_accuracy": {
                    label: float(correct[i] / support[i]) if support[i] else float("nan")
                    for i, label in enumerate(CANONICAL_CLASSES)
                },
                "support": {label: int(support[i]) for i, label in enumerate(CANONICAL_CLASSES)},
                "confusion_matrix": matrix.tolist(),
                "class_order": CANONICAL_CLASSES,
            }
        result.update({
            "selected_skill_accuracy": (
                self.selected_correct / self.selected_total if self.selected_total else float("nan")
            ),
            "selected_skill_total": self.selected_total,
            "switch_count": self.switch_count,
            "terrain_transition_detection_delay_steps": (
                float(np.mean(self.delays)) if self.delays else None
            ),
            "terrain_transition_detection_delays": self.delays,
        })
        return result


def _force_commands(env, selected_skills, fixed_command, heading_gain=0.4, max_yaw_rate=1, lookahead=5.0):
    if fixed_command is None:
        speeds = torch.tensor(
            [SKILL_SPEEDS[skill] for skill in selected_skills],
            device=env.device, dtype=env.commands.dtype,
        )
    else:
        speeds = torch.full(
            (env.num_envs,), fixed_command, device=env.device, dtype=env.commands.dtype
        )
    env.commands[:, 0] = speeds
    env.commands[:, 1] = 0.0

    # Lateral offset from track center (y only) -- never reference center_x directly
    center_y = env.simulator.env_origins[:, 1].to(env.commands.dtype)
    current_y = env.simulator.base_pos[:, 1].to(env.commands.dtype)
    lateral_error = center_y - current_y  # +y means center is to the left

    # Bearing to a lookahead point straight ahead + laterally corrected,
    # expressed relative to current heading (small-angle style, no backward flips)
    target_heading_offset = torch.atan2(lateral_error, torch.full_like(lateral_error, lookahead))

    heading_error = wrap_to_pi(target_heading_offset)
    yaw_rate_cmd = (heading_gain * heading_error).clamp(-max_yaw_rate, max_yaw_rate)
    env.commands[:, 2] = yaw_rate_cmd

    if env.commands.shape[1] > 3:
        env.commands[:, 3] = 0.0

    return lateral_error


def _depth_valid(depth):
    flattened = depth.reshape(depth.shape[0], -1)
    return torch.isfinite(flattened).all(dim=1) & (flattened.abs().sum(dim=1) > 0)


def _summary_rows(rows, track_id=None):
    subset = rows if track_id is None else [row for row in rows if row["track_id"] == track_id]
    count = len(subset)
    return {
        "track_id": track_id,
        "completed_episodes": count,
        "success_rate": float(np.mean([r["success"] for r in subset])) if count else float("nan"),
        "mean_forward_distance_m": (
            float(np.mean([r["max_forward_distance_m"] for r in subset])) if count else float("nan")
        ),
        "course_completion_count": sum(r["termination_reason"] == "course_complete" for r in subset),
        "timeout_success_count": sum(r["termination_reason"] == "timeout" for r in subset),
        "instantaneous_correct": sum(r["instantaneous_correct"] for r in subset),
        "instantaneous_total": sum(r["instantaneous_total"] for r in subset),
        "bayes_correct": sum(r["bayes_correct"] for r in subset),
        "bayes_total": sum(r["bayes_total"] for r in subset),
    }


def run_eval(args):
    if "genesis" in globals().get("SIMULATOR", ""):
        init_genesis(args, gs)
    else:
        configure_runtime_device(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    override_configs_multiterrain(env_cfg, args)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    device = env.device
    track_ids = torch.arange(env.num_envs, device=device, dtype=torch.long) % 10
    assign_tracks_and_reset(env, track_ids)
    obs_buf, _, obs_history, _, _, depth = unpack_observations(env.get_observations(), 6)

    approach = resolve_classifier_approach(args)
    runtime_classifier = CLASSIFIER_LOADERS[approach](args.classifier_dir, device)
    bayes_template, bayes_path = BAYES_FILTER_LOADERS[args.bayes_filter_approach](args, device)
    if runtime_classifier.class_ids != list(bayes_template.labels):
        raise ValueError(
            "Classifier class_ids and Bayes filter labels/order must agree exactly: "
            f"{runtime_classifier.class_ids!r} != {list(bayes_template.labels)!r}"
        )
    filters = [copy.deepcopy(bayes_template) for _ in range(env.num_envs)]
    for filt in filters:
        filt.reset()
    policy_set = PerTerrainPolicySet(args.jit, device)
    terminal_capture = TerminalCapture(env)

    selected_skills = ["rough"] * env.num_envs
    selection_initialized = [False] * env.num_envs

    from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply, quat_from_euler_xyz
    forward = quat_apply(env.simulator.base_quat, env.forward_vec)
    env.heading = torch.atan2(forward[:, 1], forward[:, 0]) 

    if args.selector_mode == "oracle":
        _, selected_skills = get_ground_truth_labels(env, args.look_ahead_frac)
    assigned_lora = torch.tensor(
        [label_to_lora(label) for label in selected_skills], device=device, dtype=torch.long
    )

    start_x = env.simulator.base_pos[:, 0].clone()
    max_progress = torch.zeros(env.num_envs, device=device)
    episode_steps = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    inst_correct = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    inst_total = torch.zeros_like(inst_correct)
    bayes_correct = torch.zeros_like(inst_correct)
    bayes_total = torch.zeros_like(inst_correct)
    completed_by_track = [0] * 10
    episode_rows = []
    classification = ClassificationStats()
    previous_truth = [None] * env.num_envs
    pending_transition = [None] * env.num_envs
    finish_x = 5 * env_cfg.terrain.terrain_length - args.finish_margin
    width = env_cfg.terrain.terrain_width
    steps_run = 0

    while steps_run < args.num_steps and min(completed_by_track) < args.episodes_per_track:
        _force_commands(env, selected_skills, args.fixed_forward_command)
        with torch.inference_mode():
            actions = policy_set.act(obs_buf, obs_history, depth, assigned_lora)
        terminal_capture.begin()
        try:
            step_value = env.step(actions.detach())
        finally:
            terminal_capture.end()
        obs_buf, _, obs_history, _, _, _, dones, infos, depth = unpack_observations(step_value, 9)
        steps_run += 1
        episode_steps += 1
        timeout_flags = infos.get("time_outs") if isinstance(infos, dict) else None

        step_positions = env.simulator.base_pos.clone()
        for env_id, state in terminal_capture.states.items():
            step_positions[env_id] = state["position"]
        finish_distance = (finish_x - start_x).clamp_min(0.0)
        progress = (step_positions[:, 0] - start_x).clamp_min(0.0)
        max_progress = torch.maximum(max_progress, torch.minimum(progress, finish_distance))

        done_ids = set(dones.nonzero(as_tuple=False).flatten().detach().cpu().tolist())
        course_ids = set((step_positions[:, 0] >= finish_x).nonzero(as_tuple=False).flatten().cpu().tolist())
        columns = track_ids.to(step_positions.device)
        lateral_failure = (
            (step_positions[:, 1] < columns * width)
            | (step_positions[:, 1] >= (columns + 1) * width)
        )
        lateral_ids = set(lateral_failure.nonzero(as_tuple=False).flatten().cpu().tolist())

        if (steps_run - 1) % args.classify_every == 0:
            raw_truth, canonical_truth = get_ground_truth_labels(env, args.look_ahead_frac)
            print(canonical_truth)
            valid = _depth_valid(depth)
            if done_ids:
                valid[list(done_ids)] = False
            valid_ids = valid.nonzero(as_tuple=False).flatten()
            if valid_ids.numel():
                sensor_depth = depth[valid_ids].detach()
                euler = env.simulator._base_euler[valid_ids].detach()
                angular_velocity = env.simulator.base_ang_vel[valid_ids].detach()
                probabilities = runtime_classifier.predict(sensor_depth, euler, angular_velocity)
                probabilities = probabilities.to(device)
                if probabilities.shape[0] != valid_ids.numel():
                    raise RuntimeError("Classifier batch size does not match valid environment count")
                for batch_index, env_id in enumerate(valid_ids.detach().cpu().tolist()):
                    truth = canonical_truth[env_id]
                    probability = probabilities[batch_index]
                    instant_id = runtime_classifier.class_ids[int(probability.argmax().item())]
                    instant_label = canonicalize_label(instant_id)
                    bayes_step = filters[env_id].update(probability)
                    bayes_label = canonicalize_label(bayes_step.label)
                    posterior = bayes_step.posterior
                    if args.selector_mode == "oracle":
                        selected = truth
                    elif args.selector_mode == "instantaneous":
                        selected = instant_label
                    elif args.selector_mode == "bayes":
                        selected = bayes_label
                    else:
                        selected = "rough"
                    if selection_initialized[env_id] and selected != selected_skills[env_id]:
                        classification.switch_count += 1
                    selection_initialized[env_id] = True
                    selected_skills[env_id] = selected
                    assigned_lora[env_id] = label_to_lora(selected)
                    inst_correct[env_id] += int(instant_label == truth)
                    inst_total[env_id] += 1
                    bayes_correct[env_id] += int(bayes_label == truth)
                    bayes_total[env_id] += 1

                    if previous_truth[env_id] is not None and truth != previous_truth[env_id]:
                        pending_transition[env_id] = (truth, steps_run)
                    previous_truth[env_id] = truth
                    pending = pending_transition[env_id]
                    if pending is not None and bayes_label == pending[0]:
                        classification.delays.append(steps_run - pending[1])
                        pending_transition[env_id] = None
                    record = {
                        "step": steps_run, "env_id": env_id,
                        "track_id": int(track_ids[env_id].item()),
                        "raw_ground_truth": raw_truth[env_id],
                        "canonical_ground_truth": truth,
                        "class_ids": [str(value) for value in runtime_classifier.class_ids],
                        "instantaneous_probabilities": probability.detach().cpu().tolist(),
                        "instantaneous_label": instant_label,
                        "bayes_posterior": posterior.detach().cpu().tolist(),
                        "bayes_label": bayes_label,
                        "selected_skill": selected,
                    }
                    classification.update(truth, instant_label, bayes_label, selected, record)

        completed_ids = sorted(done_ids | course_ids | lateral_ids)
        manual_reset_ids = []
        for env_id in completed_ids:
            track_id = int(track_ids[env_id].item())
            timeout = (
                bool(timeout_flags[env_id].item()) if timeout_flags is not None
                else terminal_capture.states.get(env_id, {}).get("timeout", False)
            )
            if env_id in course_ids:
                success, reason = True, "course_complete"
            elif timeout:
                success, reason = True, "timeout"
            elif env_id in lateral_ids:
                success, reason = False, "left_column"
            else:
                success, reason = False, "termination"
            if completed_by_track[track_id] < args.episodes_per_track:
                episode_rows.append({
                    "selector_mode": args.selector_mode,
                    "classifier_approach": approach,
                    "bayes_filter_approach": args.bayes_filter_approach,
                    "bayes_filter_path": str(bayes_path),
                    "seed": args.seed, "difficulty": args.difficulty,
                    "track_id": track_id,
                    "terrain_sequence": args.track_layout[track_id]["sequence"],
                    "success": success, "termination_reason": reason,
                    "max_forward_distance_m": float(max_progress[env_id].item()),
                    "episode_steps": terminal_capture.states.get(env_id, {}).get(
                        "episode_steps", int(episode_steps[env_id].item())
                    ),
                    "instantaneous_correct": int(inst_correct[env_id].item()),
                    "instantaneous_total": int(inst_total[env_id].item()),
                    "bayes_correct": int(bayes_correct[env_id].item()),
                    "bayes_total": int(bayes_total[env_id].item()),
                })
                completed_by_track[track_id] += 1
            if env_id not in done_ids:
                manual_reset_ids.append(env_id)

        if completed_ids:
            for env_id in completed_ids:
                filters[env_id].reset()
                selected_skills[env_id] = "rough"
                selection_initialized[env_id] = False
                previous_truth[env_id] = None
                pending_transition[env_id] = None
            if manual_reset_ids:
                reset_tensor = torch.tensor(manual_reset_ids, device=device, dtype=torch.long)
                env.reset_idx(reset_tensor)
                env.compute_observations()
                obs_buf, _, obs_history, _, _, depth = unpack_observations(env.get_observations(), 6)
            if args.selector_mode == "oracle":
                _, oracle_labels = get_ground_truth_labels(env, args.look_ahead_frac)
                for env_id in completed_ids:
                    selected_skills[env_id] = oracle_labels[env_id]
            assigned_lora = torch.tensor(
                [label_to_lora(label) for label in selected_skills], device=device, dtype=torch.long
            )
            reset_tensor = torch.tensor(completed_ids, device=device, dtype=torch.long)
            start_x[reset_tensor] = env.simulator.base_pos[reset_tensor, 0]
            max_progress[reset_tensor] = 0.0
            episode_steps[reset_tensor] = 0
            inst_correct[reset_tensor] = 0
            inst_total[reset_tensor] = 0
            bayes_correct[reset_tensor] = 0
            bayes_total[reset_tensor] = 0

    classification_summary = classification.summary()
    overall = _summary_rows(episode_rows)
    overall.update({
        "quotas_complete": all(value == args.episodes_per_track for value in completed_by_track),
        "completed_by_track": completed_by_track,
        "simulation_steps": steps_run,
        "instantaneous_classification_accuracy": classification_summary["instantaneous"]["accuracy"],
        "bayes_classification_accuracy": classification_summary["bayes"]["accuracy"],
    })
    return {
        "metadata": {
            "task": args.task, "policy_jit": args.jit,
            "selector_mode": args.selector_mode,
            "classifier_approach": approach, "classifier_dir": args.classifier_dir,
            "bayes_filter_approach": args.bayes_filter_approach,
            "bayes_filter_path": str(bayes_path), "seed": args.seed,
            "difficulty": args.difficulty, "look_ahead_frac": args.look_ahead_frac,
            "disabled_randomizations": sorted(
                name for name, enabled in EVAL_DOMAIN_RANDOMIZATION_ENABLED.items()
                if not enabled
            ),
            "track_layout": args.track_layout,
        },
        "headline": {
            "episodic_success_rate": overall["success_rate"],
            "episodic_forward_distance_m": overall["mean_forward_distance_m"],
            "instantaneous_classification_accuracy": overall["instantaneous_classification_accuracy"],
            "bayes_classification_accuracy": overall["bayes_classification_accuracy"],
        },
        "overall": overall,
        "per_column": [
            {
                **_summary_rows(episode_rows, track),
                "terrain_sequence": args.track_layout[track]["sequence"],
                "seed": args.seed,
                "difficulty": args.difficulty,
                "selector_mode": args.selector_mode,
                "classifier_approach": approach,
                "bayes_filter_approach": args.bayes_filter_approach,
            }
            for track in range(10)
        ],
        "classification": classification_summary,
        "classification_ticks": classification.ticks,
        "episodes": episode_rows,
    }


def save_results(payload, out_dir, task):
    root = Path(out_dir or Path(LEGGED_GYM_ROOT_DIR) / "exp_logs" / "classifier_eval")
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = root / f"eval_{task}_{timestamp}.json"
    csv_path = root / f"eval_{task}_{timestamp}_episodes.csv"
    with json_path.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=True)
    rows = payload["episodes"]
    with csv_path.open("w", newline="") as stream:
        if rows:
            csv_rows = [{**row, "terrain_sequence": "|".join(row["terrain_sequence"])} for row in rows]
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return json_path, csv_path


def main():
    args = get_args()
    payload = run_eval(args)
    headline = payload["headline"]
    print("\n=== Headline results ===")
    print(f"Episodic success rate:             {headline['episodic_success_rate']:.3f}")
    print(f"Mean episodic forward distance:    {headline['episodic_forward_distance_m']:.3f} m")
    print(f"Instantaneous classification acc.: {headline['instantaneous_classification_accuracy']:.3f}")
    print(f"Bayes classification accuracy:     {headline['bayes_classification_accuracy']:.3f}")
    if not payload["overall"]["quotas_complete"]:
        print(f"WARNING: incomplete track quotas: {payload['overall']['completed_by_track']}")
    for track in payload["metadata"]["track_layout"]:
        print(f"Track {track['track_id']}: {' -> '.join(track['sequence'])}")
        for cell in track["cells"]:
            print(f"  row {cell['row']}: {cell['raw_label']} {cell['parameters']}")
    json_path, csv_path = save_results(payload, args.out_dir, args.task)
    print(f"Saved JSON: {json_path}")
    print(f"Saved episodes CSV: {csv_path}")


if __name__ == "__main__":
    main()
