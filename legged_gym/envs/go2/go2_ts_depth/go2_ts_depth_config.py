from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotTSDepthCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg, get_simulator_suffix

class Go2TSDepthCfg( Go2RoughCommonCfg ):
    class env( Go2RoughCommonCfg.env ):
        num_envs = 3000
        num_camera_envs = 1000
        num_observations = 45
        num_privileged_obs = 241
        num_latent_dims = 32
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 15 + 3 + 48 + 144
        num_critic_obs = c_frame_stack * num_single_critic_obs
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_actions = 12
        debug_draw_depth_images = True
        debug_draw_height_points_around_base = True
    
    class terrain( Go2RoughCommonCfg.terrain ):
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4] # 16x9=144
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 10.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [slope, random uniform, stairs up, stairs down, discrete,
        #                 stepping stones, gaps, pit]
        terrain_proportions = [0.1, 0.0, 0.2, 0.2, 0.2,
                               0.1, 0.1, 0.1]
        terrain_curriculum_difficulty = {
            "slope": "difficulty * 0.4", # max slope in radians, will be multiplied by curriculum difficulty
            "step_height": "0.05 + 0.15 * difficulty", 
            "discrete_height": "0.05 + 0.15 * difficulty",
            "stepping_stones_size": "1.5",
            "stone_distance": "0.1 + difficulty * 0.8",
            "gap_size": "0.1 + difficulty * 0.8",
            "pit_depth": "0.1 + 0.6 * difficulty"
        }
        

    class control( Go2RoughCommonCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 30.}   # [N*m/rad]
        damping = {'joint': 0.75}     # [N*m*s/rad]

    class asset( Go2RoughCommonCfg.asset ):
        # Common: 
        contact_state_link_names = ["base", "Head", "thigh", "calf", "foot"]
        penalize_contacts_on = ["thigh", "calf", "base", "Head"]
        terminate_after_contacts_on = []

    class rewards( Go2RoughCommonCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        foot_clearance_target = 0.06 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( Go2RoughCommonCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -2.0
            collision = -5.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
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
            foot_clearance = 0.2

    class commands( Go2RoughCommonCfg.commands ):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( Go2RoughCommonCfg.commands.ranges ):
            lin_vel_x = [0.0, 0.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(Go2RoughCommonCfg.domain_rand):
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
    
    class normalization( Go2RoughCommonCfg.normalization):
        clip_actions = 20.0
    
    class sensor( Go2RoughCommonCfg.sensor ):
        add_depth = True
        use_warp = True       # whether to use warp-based model
        class depth_camera_config( Go2RoughCommonCfg.sensor.depth_camera_config ):
            num_sensors = 1
            num_history = 1        # history frames for depth images
            near_clip = 0.1
            far_clip = 2.0
            near_plane = 0.1
            far_plane = 10.0
            resolution = (60, 80)
            horizontal_fov_deg = 75
            pos =   (0.3, 0.0, 0.1)
            euler_gym = (0.0, 0.3, 0.0)
            euler = (0.0, 1.57 + 0.3, 0.0)
            decimation = 5
            # Warp only
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False

class Go2TSDepthCfgPPO( LeggedRobotTSDepthCfgPPO ):
    seed = 42
    distillation = False
    class policy( LeggedRobotTSDepthCfgPPO.policy ):
        clip_actions = Go2TSDepthCfg.normalization.clip_actions
        critic_hidden_dims = [1024, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        cnn_input_channel = Go2TSDepthCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [4, 8]
        cnn_strides = [2, 1]
        cnn_fc_layer_dims = [128]
        combination_mlp_dims = [128, 32]
        cnn_kernel_sizes = [5, 3]
        rnn_type = 'gru'
        rnn_hidden_size = 256
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotTSDepthCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        
    class runner( LeggedRobotTSDepthCfgPPO.runner ):
        teacher_model_path = "Mar09_17-57-51_/model_4500.pt"
        run_name = ''
        experiment_name = 'go2_depth'
        save_interval = 500
        load_run = "Mar18_18-01-46_"
        checkpoint = -1
        max_iterations = 10000