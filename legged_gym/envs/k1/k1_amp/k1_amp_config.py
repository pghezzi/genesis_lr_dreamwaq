from legged_gym.envs.base.legged_robot_amp_config import LeggedRobotAMPCfgPPO, LeggedRobotAMPCfg
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg
from legged_gym import SIMULATOR

import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + f"/resources/reference_motion/booster_k1/{SIMULATOR}/*")

class K1AMPCfg(LeggedRobotAMPCfg):
    class env(K1FlatCommonCfg.env):
        num_envs = 4096
        frame_stack = 5
        num_single_obs = 75 + 36
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 54
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 22
        amp_motion_files = MOTION_FILES
        max_projected_gravity = -0.3
    
    class init_state(K1FlatCommonCfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.6
    class control(K1FlatCommonCfg.control):
        pass
    class asset(K1FlatCommonCfg.asset):
        pass
    
    class rewards(K1FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.99
        base_height_target = 0.54
        foot_height_offset = 0.038
        foot_clearance_target = 0.08
        foot_distance_threshold = 0.12
        about_landing_threshold = 0.04
        only_positive_rewards = False
        class scales(K1FlatCommonCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            feet_distance = -100.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            dof_power = -1.e-4
            collision = -1.0
            action_rate = -0.01
            foot_clearance = 0.4
            foot_flat = 0.2
            feet_air_time = 1.0
            feet_contact_stand_still = 0.5
            foot_landing_vel = -0.15
    
    class domain_rand(K1FlatCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.4
        randomize_com_displacement = True
        com_pos_x_range = [-0.02, 0.02]
        com_pos_y_range = [-0.02, 0.02]
        com_pos_z_range = [-0.02, 0.02]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
    
    class commands(K1FlatCommonCfg.commands):
        curriculum = True
        max_curriculum = 1.
        resampling_time = 10.0
        heading_command = False
        zero_cmd_prob = 0.1
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.0, 0.0]    # min max [rad/s]
            heading = [0.0, 0.0]
    
    # class normalization(K1FlatCommonCfg.normalization):
    #     class obs_scales(K1FlatCommonCfg.normalization.obs_scales):
    #         key_body_pos = 1.0
    #     clip_observations = 100.
    #     clip_actions = 100.

class K1AMPCfgPPO(LeggedRobotAMPCfgPPO):
    class runner( LeggedRobotAMPCfgPPO.runner ):
        amp_reward_coef = 2.0 * K1AMPCfg.control.dt # consistent with other rewards by multiplying dt
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = K1AMPCfg.env.num_envs * LeggedRobotAMPCfgPPO.runner.num_steps_per_env * 10
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3                 # 任务奖励混合比例
        
        max_iterations = 10000
        save_interval = 200
        run_name = f'k1_amp_{SIMULATOR}'
        experiment_name = 'k1_amp'