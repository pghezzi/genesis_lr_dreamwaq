from legged_gym import *
from legged_gym.envs.base.legged_robot_dreamwaq_config import LeggedRobotDreamwaqCfg, LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg
from legged_gym.utils.terrain_vars import get_env_vars

terrain_name, finetune, terrain_index, terrain_list = get_env_vars()

import os
extra = os.environ.get("EXTRA", "")


class Go2DepthWaqCfg( LeggedRobotDreamwaqCfg ):

    if terrain_name != "baseline":
        class termination():
            reset_unrecoverable_gaps = True
            gap_terrain_depth_threshold = 1.0
            gap_foot_drop_threshold = 0.25
            gap_base_drop_threshold = 0.30
            gap_min_fallen_feet = 1
            gap_reset_steps = 4
    #else:
    #    class termination():
    #        reset_unrecoverable_gaps = False

    class env( LeggedRobotDreamwaqCfg.env ):
        num_envs = 3000
        #num_envs = 256
        num_camera_envs = 3000
        #num_camera_envs = 256
        num_actions = 12
        num_observations = 45 # 45 num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        debug_draw_height_points_around_base = True
    
    class terrain( Go2RoughCommonCfg.terrain ):
        # only Stairs, Gap, Pits, High-Platform are used for fft and lora
        # § Tall Stairs [5cm – 30 cm]
        # § Gap [0.1m - 1.0 m]
        # § Pits [0.1 m – 0.6 m]
        # § High-Platform [0.1 m – 0.6 m height]
        # difficulty ranges from 0 to (num_rows - 1) / num_rows
        terrain_curriculum_difficulty = {
            "random_uniform_params" : {
                "min_height": "-0.12",
                "max_height": "0.12",
                "step": "0.005",
                "downsampled_scale": "0.2",
            },
            "slope": "difficulty * 0.6",
            "discrete_height": "0.05 + 0.2 * difficulty",
            "stepping_stones_params": {
                "stone_length": "np.random.uniform(0.4, 1.2)",
                "stone_width": "np.random.uniform(0.4, 1.2)",
                "stone_distance_x": "0.1 + 0.7 * difficulty",
                "stone_distance_y": "np.random.uniform(0.2, 0.8)",
                "max_height": "0.20",
            },
        }
        terrain_curriculum_difficulty_custom = {
            "step_height": f"0.05 + {0.3 - 0.05} * difficulty",
            "gap_size": f"0.1 + {1 - 0.1} * difficulty",
            "pit_depth": f"0.1 + {0.6 - 0.1} * difficulty",
            "high_platform_params": {
                "high_platform_height": f"0.1 + {0.6 - 0.1} * difficulty",
                "high_platform_length": "np.random.uniform(0.6, 1.6)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_interval": "np.random.uniform(1.0, 2.0)",
            },
            "high_platform_gaps_params": {
                "high_platform_height": f"0.1 + {0.6 - 0.1} * difficulty",
                "high_platform_length": "np.random.uniform(1.6, 2.0)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_distance_y": "np.random.uniform(0.2, 2.0)",
                "gap_size": f"0.1 + {1 - 0.1} * difficulty",
            },
        }
        terrain_curriculum_difficulty.update(terrain_curriculum_difficulty_custom)
        terrain_proportions = terrain_list

    class viewer(Go2RoughCommonCfg.viewer):
        rendered_envs_idx = [0] 

    class init_state( Go2RoughCommonCfg.init_state ):
        roll_random_scale: float = 0.1  # small random roll to make the policy learn to balance in roll direction
        pitch_random_scale: float = 0.1 # small random pitch to make the policy learn to balance in pitch direction and step up/down small obstacles
        yaw_random_scale: float = 3.14  # initialize robot with random yaw to make it learn to rotate
    class control( Go2RoughCommonCfg.control ):
        # COPIED this up because I think we might need to up these gains to enable "jumping" behavior. 
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 30.}   # [N*m/rad]
        damping = {'joint': 0.75}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt = 0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
    class asset( Go2RoughCommonCfg.asset ):
        pass
    class rewards( Go2RoughCommonCfg.rewards ):
        if terrain_name == "baseline":
            class scales(Go2RoughCommonCfg.rewards.scales):
                feet_near_edge = 0
        else:
            soft_dof_pos_limit = 0.9
            base_height_target = 0.4
            foot_clearance_target = 0.08 # desired foot clearance above ground [m]
            foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
            foot_clearance_tracking_sigma = 0.01
            tracking_sigma = 0.2
            only_positive_rewards = True
            feet_edge_threshold = 0.05 # distance threshold below which foot is considered to be near the edge of a terrain
            class scales:
                # limitation
                dof_pos_limits = -2.0
                collision = -10.0
                
                # command tracking
                tracking_lin_vel = 1.5
                tracking_ang_vel = 1.0
                
                # smooth
                lin_vel_z = -1.0
                ang_vel_xy = -0.05
                orientation = -1.0
                dof_power = -2.e-5
                dof_acc = -2.e-7
                action_rate = -0.01
                action_smoothness = -0.01
                
                # gait
                hip_pos = -0.15
                foot_clearance = 0.6   # I've found making this larger tends to help with "stepping" behavior
                feet_stumble = -1.0
                feet_contact_stand_still = 0.1
                feet_near_edge = -1.0
                feet_air_time = 0.6    # this is actually used in ts_depth via inhereitence from the base class.


    class commands( LeggedRobotDreamwaqCfg.commands ):
        curriculum = True
        if terrain_name == "baseline":
            max_curriculum = 1.0
        else:
            max_curriculum = 1.5
        
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        
        if terrain_name == "baseline":
            zero_cmd_prob = 0.1
        class ranges( LeggedRobotDreamwaqCfg.commands.ranges ):
            if terrain_name == "baseline":
                lin_vel_x = [-0.5, 0.5] # min max [m/s]
                lin_vel_y = [-1.0, 1.0]   # min max [m/s]
                ang_vel_yaw = [-1, 1]    # min max [rad/s]
                heading = [-3.14, 3.14]
            else:
                lin_vel_x = [0.0, 0.5]   # min max [m/s]
                lin_vel_y = [0.0, 0.0]   # min max [m/s]
                ang_vel_yaw = [-1, 1]    # min max [rad/s]
                heading = [0.0, 0.0]
            
    class domain_rand(LeggedRobotDreamwaqCfg.domain_rand):
        if terrain_name == "baseline":
            randomize_friction = True
            friction_range = [0.2, 1.7]
            randomize_base_mass = True
            added_mass_range = [-1., 1.]
            push_robots = True
            push_interval_s = 10
            max_push_vel_xy = 1.
            randomize_com_displacement = True
            com_pos_x_range = [-0.03, 0.03]
            com_pos_y_range = [-0.03, 0.03]
            com_pos_z_range = [-0.03, 0.03]
            randomize_pd_gain = True
            kp_range = [0.8, 1.2]
            kd_range = [0.8, 1.2]
            randomize_joint_armature = False
            joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
            randomize_joint_friction = False
            joint_friction_range = [0.01, 0.02]
            randomize_joint_damping = False
            joint_damping_range = [0.25, 0.3]
        else:
            randomize_friction = True
            friction_range = [0.2, 1.7]
            randomize_base_mass = True
            added_mass_range = [-1., 2.]
            push_robots = True
            push_interval_s = 3
            max_push_vel_xy = 0.5
            randomize_com_displacement = True
            com_pos_x_range = [-0.03, 0.03]
            com_pos_y_range = [-0.03, 0.03]
            com_pos_z_range = [-0.03, 0.03]
            randomize_pd_gain = True
            kp_range = [0.8, 1.2]
            kd_range = [0.8, 1.2]
        randomize_camera_pos = True
        camera_com_displacement_range = [0.01, 0.0025, 0.03]
        randomize_camera_euler = True
        camera_euler_offset_range = [0.0577, 0.0173, 0.0577]

    class normalization( LeggedRobotDreamwaqCfg.normalization):
        class obs_scales( LeggedRobotDreamwaqCfg.normalization.obs_scales ):
            actions = 0.1
        clip_actions = 10.0
        

    class sensor(LeggedRobotDreamwaqCfg.sensor):
        add_depth = True
        use_warp = True
        class depth_camera_config(LeggedRobotDreamwaqCfg.sensor.depth_camera_config):
            decimation = 5
            resolution = [int(480/4), int(640/4)]
            num_history = 1
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
            horizontal_fov_deg = 88
            pos = (0.3, 0.0, 0.1)
            euler = (0.0, 1.57 + 0.3, 0.0)
            near_plane = 0.05
            far_plane = 4.00
            near_clip = 0.00
            far_clip = 3.00
            latency_range = [0.08, 0.142]
            latency_resampling_time = 5.0
            refresh_duration = 1/10 # [s]
            crop_top_bottom = [int(48/4), 0]
            crop_left_right = [int(28/4), int(36/4)]
            resized_resolution = [48, 64]

            #stereo_min_distance = 0.175 # when using (480, 640) resolution
            #stereo_far_distance = 1.2
            #stereo_far_noise_std = 0.08 
            #stereo_near_noise_std = 0.02
            #stereo_full_block_artifacts_prob = 0.008
            #stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
            #stereo_full_block_height_mean_std = [62, 1.5]
            #stereo_full_block_width_mean_std = [3, 0.01]
            #stereo_half_block_spark_prob = 0.02
            #stereo_half_block_value = 3000
            #sky_artifacts_prob = 0.001
            #sky_artifacts_far_distance = 2.
            #sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            #sky_artifacts_height_mean_std = [2, 3.2]
            #sky_artifacts_width_mean_std = [2, 3.2]


class Go2DepthWaqCfgPPO( LeggedRobotDreamwaqCfgPPO ):
    seed = 42
    runner_class_name = "DreamWaQDepthRunner"
    class policy( LeggedRobotDreamwaqCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
        depth_image_resolution = Go2DepthWaqCfg.sensor.depth_camera_config.resized_resolution
        cnn_input_channel = Go2DepthWaqCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [8, 8]
        cnn_strides = [1, 1]
        cnn_fc_layer_dims = [128, 64]
        cnn_kernel_sizes = [5, 3]
    class algorithm( LeggedRobotDreamwaqCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
    class runner( LeggedRobotDreamwaqCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQDepth"
        algorithm_class_name = "PPO_DreamWaQ_Depth"
        run_name = 'dreamwaq'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"
        experiment_name = f"go2_depth_waq{'_fft' if finetune else ''}{'_' + terrain_name}{'_' + extra if extra else ''}"
        pre_trained = finetune if finetune else None
        save_interval = 500
        if terrain_name == "baseline":
            max_iterations = 3000
        else:
            max_iterations = 5000
