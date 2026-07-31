from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import (
    Go2DepthWaqCfg,
    Go2DepthWaqCfgPPO,
)

# This file is located at:
#
# Legged_Gym_EX/
# └── legged_gym/
#     └── envs/
#         └── go2/
#             └── go2_depth_waq_distill/
#                 └── go2_depth_waq_distill_config.py
#
# parents[4] therefore points to the Legged_Gym_EX repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]

# Load environment variables from:
#
# Legged_Gym_EX/.env
#
# override=False means values already exported in the shell take priority
# over values written in the .env file.
load_dotenv(
    dotenv_path=REPO_ROOT / ".env",
    override=False,
)


def required_env(name: str) -> str:
    """Read a required environment variable.

    Raises a clear error during configuration loading when the variable has not
    been provided through the shell or the repository's local .env file.
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Required environment variable {name!r} is not set.\n"
            f"Create the local file:\n"
            f"  {REPO_ROOT / '.env'}\n"
            f"using the committed .env.example file."
        )

    return value.strip()


def optional_int_env(name: str, default: int) -> int:
    """Read an optional integer environment variable."""
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return int(default)

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name!r} must be an integer, "
            f"but received {raw_value!r}."
        ) from exc



class Go2DepthWaqDistillCfg(Go2DepthWaqCfg):
    """Environment configuration for multi-teacher LoRA distillation."""

    class distillation:
        # ------------------------------------------------------------------
        # Shared LoRA teacher defaults
        # ------------------------------------------------------------------
        #
        # All teachers use this baseline unless a teacher dictionary supplies
        # its own "base_model" override.
        base_model = required_env("DISTILL_BASE_MODEL")

        # All LoRA components use this rank unless a teacher dictionary
        # overrides "rank" or a component-specific rank.
        lora_rank = optional_int_env(
            "DISTILL_LORA_RANK",
            default=8,
        )

        # ------------------------------------------------------------------
        # LoRA teachers
        # ------------------------------------------------------------------
        #
        # List order defines the numeric teacher ID:
        #
        #   teachers[0] -> teacher ID 0 -> gap
        #   teachers[1] -> teacher ID 1 -> stairs
        #
        # The checkpoint paths are stored in the local .env file rather than
        # directly in this committed configuration file.
        teachers = [
            {
                "name": "gap",
                "checkpoint": required_env(
                    "DISTILL_GAP_CHECKPOINT"
                ),
            },
            {
                "name": "stairs",
                "checkpoint": required_env(
                    "DISTILL_STAIRS_CHECKPOINT"
                ),
            },
        ]

        # ------------------------------------------------------------------
        # Terrain column -> teacher mapping
        # ------------------------------------------------------------------
        #
        # Index:
        #   Genesis terrain type/column ID
        #
        # Value:
        #   index into the teachers list above
        #
        # Example:
        #
        #   terrain_type_to_teacher = [
        #       0, 0, 0,  # terrain columns 0-2 use the gap teacher
        #       1, 1, 1,  # terrain columns 3-5 use the stairs teacher
        #   ]
        #
        # You must replace this with the actual terrain-column ordering used
        # by the distillation terrain configuration.
        terrain_type_to_teacher = [
            0,
            1,
        ]

        # ------------------------------------------------------------------
        # Pure imitation-learning settings
        # ------------------------------------------------------------------
        #
        # Supported targets in the supplied PPO_WAQ_Distill scaffold:
        #
        #   "l1"      -> sum absolute action error per sample
        #   "mse"     -> mean squared action error per sample
        #   "mse_sum" -> summed squared action error per sample
        #   "l2"      -> Euclidean action-vector distance
        distill_target = "l1"

        distillation_loss_coef = 1.0

        learning_rate = 1.0e-4
        weight_decay = 0.0

        # Train the student's history encoder/VAE along with its actor.
        train_vae = True

        # Train the student's depth-image encoder along with its actor.
        train_visual_encoder = True


class Go2DepthWaqDistillCfgPPO(Go2DepthWaqCfgPPO):
    """Training configuration for pure multi-teacher distillation.

    The class inherits the repository's PPO configuration layout for
    compatibility with TaskRegistry and OnPolicyRunner. PPO_WAQ_Distill ignores
    PPO-only fields such as clipping, entropy, gamma, lambda, and value loss.
    """

    runner_class_name = "DreamWaQDepthDistillRunner"

    class algorithm(Go2DepthWaqCfgPPO.algorithm):
        learning_rate = (
            Go2DepthWaqDistillCfg.distillation.learning_rate
        )

        weight_decay = (
            Go2DepthWaqDistillCfg.distillation.weight_decay
        )

        distill_target = (
            Go2DepthWaqDistillCfg.distillation.distill_target
        )

        distillation_loss_coef = (
            Go2DepthWaqDistillCfg
            .distillation
            .distillation_loss_coef
        )

        train_vae = (
            Go2DepthWaqDistillCfg.distillation.train_vae
        )

        train_visual_encoder = (
            Go2DepthWaqDistillCfg
            .distillation
            .train_visual_encoder
        )

        # These are still used by the pure-imitation optimizer.
        num_learning_epochs = 1
        num_mini_batches = 4
        max_grad_norm = 1.0

    class runner(Go2DepthWaqCfgPPO.runner):
        # The final generalist student is an ordinary depth DreamWaQ policy.
        policy_class_name = "ActorCriticDreamWaQDepth"

        # This custom class performs pure action imitation, despite retaining
        # "PPO" in its name for repository compatibility.
        algorithm_class_name = "PPO_WAQ_Distill"

        experiment_name = (
            "go2_depth_waq_multiteacher_lora_distill"
        )
        run_name = "pure_imitation"

        max_iterations = 10000
        save_interval = 500

        # Set this only if you later want to initialize the student from an
        # existing generalist checkpoint. The LoRA teacher checkpoints are
        # configured separately above.
        pre_trained = None
