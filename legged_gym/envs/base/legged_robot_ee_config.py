from .legged_robot_config import *

class LeggedRobotEECfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        # Here the privileged_obs is actually critic_obs
        num_single_obs = 45
        frame_stack = 10    # number of frames to stack for obs_history
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 24
        c_frame_stack = 5
        single_critic_obs_len = num_single_obs + 31 + 81 + 17
        num_privileged_obs = c_frame_stack * single_critic_obs_len

class LeggedRobotEECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'EERunner' # Explicit Estimator Runner
    class policy( LeggedRobotCfgPPO.policy ):
        estimator_hidden_dims = [256, 128]
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for estimator training
        estimator_lr = 2.e-4
        num_estimator_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticEE'
        algorithm_class_name = 'PPO_EE'