from .legged_robot_config import *

class LeggedRobotDreamwaqCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 45  # num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotDreamwaqCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner" # DreamWaQ Runner
    class policy( LeggedRobotCfgPPO.policy ):
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for vae training
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"