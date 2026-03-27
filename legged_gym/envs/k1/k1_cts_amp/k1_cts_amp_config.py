from legged_gym.envs.base.template_cfgs import LeggedRobotAMPCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg
from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, K1AMPCfgPPO
from legged_gym import SIMULATOR

import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + f"/resources/reference_motion/booster_k1/{SIMULATOR}/*")

class K1_CTS_AMPCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_envs = 4096
        num_teacher = int(num_envs // 4 * 3)
        frame_stack = 5
        num_observations = 75
        num_history_obs = int(num_observations * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 70
        num_critic_obs = int(c_frame_stack * num_single_critic_obs)
        num_privileged_obs = 70
        num_latent_dims = 32
        
        num_actions = 22
        amp_motion_files = MOTION_FILES
        max_projected_gravity = -0.3
    
    class init_state(K1AMPCfg.init_state):
        pass
    class rewards(K1AMPCfg.rewards):
        pass
        class scales(K1AMPCfg.rewards.scales):
            pass
    
    class domain_rand(K1AMPCfg.domain_rand):
        pass
    
    class commands(K1AMPCfg.commands):
        pass

class K1_CTS_AMPCfgPPO(K1AMPCfgPPO):
    runner_class_name = 'CTS_AMP_Runner'
    class policy( K1AMPCfgPPO.policy ):
        critic_hidden_dims = [1024, 512, 256]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]       # for MLP
    class algorithm(K1AMPCfgPPO.algorithm):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1
    class runner( K1AMPCfgPPO.runner ):
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'PPO_CTS_AMP'
        
        max_iterations = 30000
        save_interval = 200
        run_name = f'k1_cts_amp_{SIMULATOR}'
        experiment_name = 'k1_amp'