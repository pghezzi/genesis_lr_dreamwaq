from .legged_robot_config import *

class LeggedRobotTSDepthCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        single_critic_obs_len = num_observations
        num_critic_obs = c_frame_stack * single_critic_obs_len

class LeggedRobotTSDepthCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'TSRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP" # "MLP" or "TCN"
        history_encoder_hidden_dims = [256, 128]       # for MLP
        history_encoder_channel_dims = [1, 1, 1, 1]    # for TCN
        history_encoder_dilation = [1, 1, 2, 1]        # for TCN
        history_encoder_stride = [1, 2, 1, 2]          # for TCN
        history_encoder_final_layer_dim = 128          # for TCN
        kernel_size = 5
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticTS'
        algorithm_class_name = 'PPO_TS'