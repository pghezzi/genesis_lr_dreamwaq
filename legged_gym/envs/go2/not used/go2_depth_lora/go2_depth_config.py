from legged_gym import *
from legged_gym.envs.base.legged_robot_dreamwaq_config import LeggedRobotDreamwaqCfg, LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg


import os

TERRAIN_KEYS = [
    "rough",
    "slope",
    "stairs",
    "discrete",
    "wave",
    "stepping_stones",
]

TERRAIN_MAP = {
    name: [1 if i == idx else 0 for i in range(len(TERRAIN_KEYS))]
    for idx, name in enumerate(TERRAIN_KEYS)
}

terrain_name = os.environ.get("TERRAIN", "rough").lower()
finetune = os.environ.get("FINETUNE", "")


if terrain_name not in TERRAIN_MAP:
    raise ValueError(f"Unknown TERRAIN '{terrain_name}'. Valid options: {TERRAIN_KEYS}")

terrain_index = TERRAIN_KEYS.index(terrain_name)
terrain_list = TERRAIN_MAP[terrain_name]

experiment_extra = f"exp{terrain_index+1}"

class Go2DepthCfg( LeggedRobotDreamwaqCfg ):
    class env( LeggedRobotDreamwaqCfg.env ):
        num_envs = 256
        num_actions = 12
        num_observations = 3117 # 45 num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        c_frame_stack = 5
        single_critic_obs_len = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * single_critic_obs_len
        debug_draw_height_points_around_base = True
    
    class terrain( Go2RoughCommonCfg.terrain ):
        #mesh_type = "plane"
        terrain_proportions = terrain_list
        pass
    class init_state( Go2RoughCommonCfg.init_state ):
        pass
    class control( Go2RoughCommonCfg.control ):
        pass
    class asset( Go2RoughCommonCfg.asset ):
        pass
    class rewards( Go2RoughCommonCfg.rewards ):
        class scales( Go2RoughCommonCfg.rewards.scales ):
            pass

    class commands( LeggedRobotDreamwaqCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotDreamwaqCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(LeggedRobotDreamwaqCfg.domain_rand):
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

    class sensor:
        add_depth = True
        use_warp = True
        num_sensors = 1
        class depth_camera_config:
            return_pointcloud = False
            segmentation_camera = False
            resolution = [int(480/4), int(640/4)]
            num_history = 1
            calculate_depth = True
            horizontal_fov_deg = 88
            pos = (0.32, 0.0, 0.07)
            pos_std = [0.01, 0.0025, 0.03]
            #pos_std = [1.00, 0.0025, 0.03] #exaggerated for testing
            euler = (0, 0, 0)
            euler_std = [0.0577, 0.0173, 0.0577]
            #euler_std = [1.5, 0.5, 1.5] #exaggerated for testing
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

            stereo_min_distance = 0.175 # when using (480, 640) resolution
            stereo_far_distance = 1.2
            stereo_far_noise_std = 0.08 
            stereo_near_noise_std = 0.02
            stereo_full_block_artifacts_prob = 0.008
            stereo_full_block_values = [0.0, 0.25, 0.5, 1., 3.]
            stereo_full_block_height_mean_std = [62, 1.5]
            stereo_full_block_width_mean_std = [3, 0.01]
            stereo_half_block_spark_prob = 0.02
            stereo_half_block_value = 3000
            sky_artifacts_prob = 0.001
            sky_artifacts_far_distance = 2.
            sky_artifacts_values = [0.6, 1., 1.2, 1.5, 1.8]
            sky_artifacts_height_mean_std = [2, 3.2]
            sky_artifacts_width_mean_std = [2, 3.2]

#  class sensor( Go2FieldCfg.sensor ):
#         class forward_camera:
#             obs_components = ["forward_depth"]
#             resolution = [int(480/4), int(640/4)]
#             position = dict(
#                 mean= [0.24, -0.0175, 0.12],
#                 std= [0.01, 0.0025, 0.03],
#             )
#             rotation = dict(
#                 lower= [-0.1, 0.37, -0.1],
#                 upper= [0.1, 0.43, 0.1],
#             )
#             resized_resolution = [48, 64]
#             output_resolution = [48, 64]
#             horizontal_fov = [86, 90]
#             crop_top_bottom = [int(48/4), 0]
#             crop_left_right = [int(28/4), int(36/4)]

class Go2DepthCfgPPO( LeggedRobotDreamwaqCfgPPO ):
    class policy( LeggedRobotDreamwaqCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotDreamwaqCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
    class runner( LeggedRobotDreamwaqCfgPPO.runner ):
        run_name = 'dreamwaq'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        elif SIMULATOR == "isaaclab":
            run_name += "_isaaclab"
        experiment_name = f'go2_depth_{experiment_extra}'
        pre_trained = finetune if finetune else None
        save_interval = 500
        max_iterations = 3000