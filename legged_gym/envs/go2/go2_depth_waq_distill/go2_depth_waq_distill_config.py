from __future__ import annotations

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import (
    Go2DepthWaqCfg,
    Go2DepthWaqCfgPPO,
)


class Go2DepthWaqDistillCfg(Go2DepthWaqCfg):
    class distillation:
        # Each entry reconstructs the repo's EXISTING
        # ActorCriticDreamWaQDepthLora and then loads that skill checkpoint.
        #
        # `base_model` is the non-LoRA baseline used when that teacher was
        # originally created. `checkpoint` is the saved specialized LoRA model.
        teachers = [
            # {
            #     "name": "stairs",
            #     "base_model": "/abs/path/base_depth_waq/model_5000.pt",
            #     "checkpoint": "/abs/path/lora_stairs/model_3000.pt",
            #     "rank": 8,
            # },
            # {
            #     "name": "gaps",
            #     "base_model": "/abs/path/base_depth_waq/model_5000.pt",
            #     "checkpoint": "/abs/path/lora_gaps/model_3000.pt",
            #     "rank": 8,
            # },
        ]

        # Index = Genesis `_terrain_types` value (terrain column).
        # Value = index into `teachers`.
        #
        # Example:
        # terrain_type_to_teacher = [0, 0, 0, 1, 1, 1]
        #
        # None means teacher ID == terrain type.
        terrain_type_to_teacher = None

        distill_target = "l1"  # l1, mse, mse_sum, or l2
        distillation_loss_coef = 1.0
        learning_rate = 1.0e-4
        weight_decay = 0.0
        train_vae = True
        train_visual_encoder = True


class Go2DepthWaqDistillCfgPPO(Go2DepthWaqCfgPPO):
    runner_class_name = "DreamWaQDepthDistillRunner"

    class algorithm(Go2DepthWaqCfgPPO.algorithm):
        # Only these fields are consumed by PPO_WAQ_Distill. Inherited PPO
        # fields are accepted and ignored so the config remains compatible.
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
            Go2DepthWaqDistillCfg.distillation.distillation_loss_coef
        )
        train_vae = Go2DepthWaqDistillCfg.distillation.train_vae
        train_visual_encoder = (
            Go2DepthWaqDistillCfg.distillation.train_visual_encoder
        )

    class runner(Go2DepthWaqCfgPPO.runner):
        # The generalist student remains the repo's ordinary depth DreamWaQ.
        # Only the fixed teachers are LoRA policies.
        policy_class_name = "ActorCriticDreamWaQDepth"
        algorithm_class_name = "PPO_WAQ_Distill"
        experiment_name = "go2_depth_waq_multiteacher_lora_distill"
        run_name = "pure_imitation"
        pre_trained = None