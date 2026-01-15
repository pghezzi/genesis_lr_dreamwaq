from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1PosCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 67 # robot_state + other privilged info + terrain_heights (121)
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 5
        grf_dim = 12
        whole_body_dim = 18
        debug = False # if debugging, visualize contacts, 
        debug_viz = False # draw debug visualizations

    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.
        
    # class terrain( LeggedRobotCfg.terrain ):
    #     mesh_type = "heightfield" # none, plane, heightfield or trimesh
    #     horizontal_scale = 0.2 # [m]. if use smaller horizontal scale, need to decrease terrain_length and terrain_width, or it will compile very slowly.
    #     vertical_scale = 0.005 # [m]
    #     border_size = 5 # [m]. implemented a out_of_bound detection, so border_size can be smaller
    #     curriculum = True
    #     friction = 1.0
    #     restitution = 0.
    #     # rough terrain only:
    #     measure_heights = True
    #     measured_points_x = [-1.0, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
    #     measured_points_y = [-1.0, -0.8, -0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
    #     # measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    #     # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    #     selected = False # select a unique terrain type and pass all arguments
    #     terrain_kwargs = None # Dict of arguments for selected terrain
    #     max_init_terrain_level = 1 # starting curriculum state
    #     terrain_length = 6.0 # 
    #     terrain_width = 6.0  # 
    #     num_rows = 8  # number of terrain rows (levels)
    #     num_cols = 5  # number of terrain cols (types)
    #     # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    #     terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2]
    #     # trimesh only:
    #     slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state( LeggedRobotCfg.init_state ):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,     # [rad]
            'RL_hip_joint': 0.0,     # [rad]
            'FR_hip_joint': 0.0 ,    # [rad]
            'RR_hip_joint': 0.0,     # [rad]

            'FL_thigh_joint': 0.8,   # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'FR_thigh_joint': 0.8,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }
        # initial state randomization
        yaw_angle_range = [0., 3.14] # min max [rad]

    class normalization (LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            grf = 0.01
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 50.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.8]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        max_push_torque = 0.5
        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = False
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_stiffness = False
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = False
        joint_damping_range = [0.25, 0.3]

        enable_additional_ratio = 0.5


    class noise (LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        rendered_envs_idx = [i for i in range(0, 3, 1)]  # number of environments to be rendered
        # rendered_envs_idx.extend([i for i in range(200, 203, 1)])  # number of environments to be rendered
        # # rendered_envs_idx.extend([i for i in range(500, 503, 1)])  # number of environments to be rendered
        # # rendered_envs_idx.extend([i for i in range(750, 753, 1)])  # number of environments to be rendered
        # rendered_envs_idx.extend([i for i in range(900, 903, 1)])  # number of environments to be rendered

        # rendered_envs_idx.extend([i for i in range(1500, 1503, 1)])
        # # rendered_envs_idx.extend([i for i in range(1900, 1903, 1)])
        # # rendered_envs_idx.extend([i for i in range(3500, 3503, 1)])
        # rendered_envs_idx.extend([i for i in range(4000, 4003, 1)])

        # rendered_envs_idx.extend([i for i in range(1700, 1703, 1)])
        # # rendered_envs_idx.extend([i for i in range(2200, 2203, 1)])
        # # rendered_envs_idx.extend([i for i in range(3700, 3703, 1)])
        # rendered_envs_idx.extend([i for i in range(3900, 3903, 1)])
        # rendered_envs_idx = [0, 1000, 3500]
        add_camera = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        dof_names = [        # specify the sequence of actions
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',]
        foot_name = ["foot"]
        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        links_to_keep = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        self_collisions = True
  
    # self.joint_max = [1.047, 2.966, -0.837]
    # self.joint_min = [-1.047, -0.633, -2.721]  
    #                [0.0, 0.8, -1.5]
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        # Much smaller values than typical... only used for feedback control
        stiffness = {'joint': 30.0}   # [N*m/rad]
        damping   = {'joint': 0.75}     # [N*m*s/rad]
        
        action_scale = [0.25, 0.25, 0.25]    # action scale: target angle = action_scale * pose_action + defaultAngle        
        
        dt =  0.01     # control frequency 100Hz
        decimation = 5  # decimation: Number of control action updates @ sim DT per policy DT


    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 1.00  # [rad] ~ 40 degrees
        pitch_threshold   = 0.70  # [rad] ~ 30 degrees
        height_min = 0.20       # [m]
        height_max = 1.50        # [m]

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.26
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        
        foot_clearance_target = 0.060 # desired foot clearance above ground [m]
        foot_height_offset = 0.022    # height of the foot coordinate origin above ground [m]
        
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True

        use_reward_curriculum = False

        max_contact_force = 200.0
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination      = 0.0
            collision        = -1.0
            dof_pos_limits        = -5.0
            dof_close_to_default  = -0.05
            torque_limits         = -1.0
            
            no_motion_penalty     = 0.0
            alive_bonus           = 0.1

            stand_still_contact = -0.01
            stand_still         = -0.1

            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            dof_tracking      = 0.25
            sparse_contacts   = 0.05
            foot_swing  = 0.00
            
            # smoothness and stability
            lin_vel_z        = -1.0
            base_height      = -1.0
            ang_vel_xy       = -0.05
            orientation      = -1.0
            dof_acc          = -2.5e-7
            joint_power      = -2.e-5
            joint_power_dist = -1.e-5
            torques          = 0.0     # don't need to use this when we already have joint power above...

            # Zero out some values that are used in the individual reward classes below
            action_rate       = -0.01
            action_smoothness = -0.01

            # gait
            feet_air_time    = 0.5            # tracking reward for long steps
            max_contact_time = 0.5            # penalty for feet being in contact for too long
            foot_clearance   = 0.5            # tracking reward for feet reaching the desired clearance
            foot_slip        = -0.1           # penalty for feet slipping
            feet_contact_forces = -1.0e-1     # penalty for high contact forces on the feet
            raibert  = 0.01                    # tracking reward foot placement in x/y-plane

        class reward_curriculum():
            # curr_reward_keys = ["tau_action_rate", "tau_action_smoothness", "feedforward_torques",
            #                     "pos_action_rate", "pos_action_smoothness", "feedback_torques",
            #                     "dof_close_to_default", "dof_acc", "joint_power", "joint_power_dist",
            #                     "stable_grf_dynamics", "floating_base_stability", "raibert", 
            #                     "feet_contact_forces"]
            #                     # "wb_dynamics"]

            # curr_reward_keys = ["tau_action_rate", "tau_action_smoothness",
            #                     "pos_action_rate", "pos_action_smoothness",
            #                     "dof_close_to_default", "dof_acc", "joint_power", "joint_power_dist",
            #                     "raibert", "feet_contact_forces", "feedback_torques", "feedforward_torques"]

            # curr_reward_keys = ["tau_action_rate", "tau_action_smoothness", "feedforward_torques",
            #                     "pos_action_rate", "pos_action_smoothness", "feedback_torques",
            #                     "dof_close_to_default", "dof_acc", "joint_power", "joint_power_dist"]
            #                     # "wb_dynamics"]

            curr_reward_keys = ["tau_action_rate", "tau_action_smoothness",
                                "pos_action_rate", "pos_action_smoothness",
                                "dof_acc", "joint_power", "joint_power_dist",
                                "feet_contact_forces", 
                                "feedback_torques", "feedforward_torques"]
            
            curr_reward_bounds = {"tau_action_rate":[-1.0e-10, -1.0e-2],
                                  "tau_action_smoothness":[-1.0e-10, -1.0e-2],
                                  
                                  "pos_action_rate":[-1.0e-10, -1.0e-2],
                                  "pos_action_smoothness":[-1.0e-10, -1.0e-2],
                                  
                                  "dof_acc":[-1.0e-12, -2.5e-7],
                                  
                                  "joint_power":[-2.0e-10, -2.0e-5],
                                  "joint_power_dist":[-1.0e-10, -1.0e-5],
                                  
                                  "feedback_torques":[-2.0e-8, -2.0e-4],
                                  "feedforward_torques":[-2.0e-8, -2.0e-4],

                                  "feet_contact_forces":[-1.0e-10,-1e-1]
                                 }

            curr_steps = 8000
            warmup_steps = 0

    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class GO1PosCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "OnPolicyRunnerPos" # Teacher-Student Runner
    
    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'tanh' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, swish (SiLU)
        init_noise_std = 1.00
        
        # Context encoder
        cenet_enc_layers=[128,64]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 3

        # Context Decoder
        cenet_dec_input_dim = 19
        cenet_dec_layers = [64,128]
        cenet_dec_out_dim = 45        # next obs (57) + grf_dim (12)

        # Actor/critic
        actor_shared_dim = 512
        actor_branch_layers = [256,128]
        critic_layers = [256,128,64]
        
        # Shared
        dropout = 0.1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.0e-3 #
        # learning_rate = 3.0e-4 #
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam   = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_Pos'
        algorithm_class_name = 'PPOPos'
        num_steps_per_env = 144 # per iteration
        max_iterations = 1500 # number of policy updates
        grf_dim = 12
        
        # debug_warmpinn_wb
        run_name = '100hz_pos_baseline_01'
        experiment_name = 'rss_go1_pos'
        save_interval = 100
        
        
        load_run = "Jan14_13-48-38_100hz_pos_baseline_01"
        checkpoint = 1500

        # Load parameters for first function policy
        # run_name = 'test_01'
        # experiment_name = 'go1_dynamic'
        # save_interval = 50
        # load_run = "Dec01_18-26-31_test_01"
        # checkpoint = 3000
