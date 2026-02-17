from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1DynamicWaterCfg( LeggedRobotCfg ):

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.

    class init_state( LeggedRobotCfg.init_state ):
        leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                            [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
        rot = [1.0, 0.0, 0.0, 0.0] # w, x, y, z [quat]
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
        
        # These have been checked using the BalanceStand controller that leverages the WBIC impulse control loop. 
        #   The standing posture assumed by this control is about only a few fractions of a radian off from the above, so should be fine
        # default_joint_torques = { # = target joint torques [nM] when action = 0.0
        #     'FR_hip_joint':  1.5,   # [nM]
        #     'FL_hip_joint': -1.5,   # [nM]
        #     'RR_hip_joint':  1.5,   # [nM]
        #     'RL_hip_joint': -1.5,   # [nM]

        #     'FL_thigh_joint': 0.5,  # [nM]
        #     'RL_thigh_joint': 0.5,  # [nM]
        #     'FR_thigh_joint': 0.5,  # [nM]
        #     'RR_thigh_joint': 0.5,  # [nM]

        #     'FL_calf_joint': 4.0,   # [nM]
        #     'RL_calf_joint': 4.0,   # [nM]
        #     'FR_calf_joint': 5.0,   # [nM]
        #     'RR_calf_joint': 5.0,   # [nM]
        # }

        default_joint_torques = { # = target joint torques [nM] when action = 0.0
            'FR_hip_joint':  0.0,   # [nM]
            'FL_hip_joint':  0.0,   # [nM]
            'RR_hip_joint':  0.0,   # [nM]
            'RL_hip_joint':  0.0,   # [nM]

            'FL_thigh_joint': 0.0,  # [nM]
            'RL_thigh_joint': 0.0,  # [nM]
            'FR_thigh_joint': 0.0,  # [nM]
            'RR_thigh_joint': 0.0,  # [nM]

            'FL_calf_joint': 0.0,   # [nM]
            'RL_calf_joint': 0.0,   # [nM]
            'FR_calf_joint': 0.0,   # [nM]
            'RR_calf_joint': 0.0,   # [nM]
        }
        # initial state randomization
        yaw_angle_range = [0., 3.14] # min max [rad]

    class normalization (LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_tau = 0.05               # in collected data the magnitude of the DOF's velocity and torques are roughly comparable 
            grf = 0.01
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 50.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.8]

        # What changes with finetuning round
        # push_robots = False
        push_robots = True
        
        push_interval_max = 15.0
        push_interval_min = 1.0
        max_push_vel_xy = 1.00
        min_push_vel_xy = 1.00

        max_vertical_push = -0.5
        min_vertical_push = 0.00
        vert_interval_max = 10.0
        vert_interval_min = 0.1

        max_push_torque = 2.5
        min_push_torque = 0.75
        wrench_timeout_min = 0.01
        wrench_timeout_max = 5.0
        
        num_push_steps = 500  # number of steps to apply the same randomization
        push_warmup = 3000
        
        randomize_base_mass = False
        # added_mass_range = [-1.0, 8.0]
        min_added_mass_max = 2.0
        max_added_mass_max = 8.0
        added_mass_min = -1.0
        
        randomize_com_displacement = False
        # com_displacement_range_xy = [-0.25, 0.25]
        com_displacement_range_z = [0.10, 0.30]

        com_displacement_xy_min = 0.075
        com_displacement_xy_max = 0.25
        
        # # What changes with finetuning round
        # push_robots = True
        # push_interval_max = 15.0
        # push_interval_min = 8.0
        # max_push_vel_xy = 2.00
        # min_push_vel_xy = 0.5

        # max_vertical_push = -0.5
        # min_vertical_push = 0.00
        # vert_interval_max = 10.0
        # vert_interval_min = 0.1

        # max_push_torque = 2.5
        # min_push_torque = 0.75
        # wrench_timeout_min = 8.0
        # wrench_timeout_max = 5.0
        
        # num_push_steps = 1  # number of steps to apply the same randomization
        # push_warmup = 0
        
        # randomize_base_mass = True
        # # added_mass_range = [-1.0, 8.0]
        # min_added_mass_max = 6.0
        # max_added_mass_max = 8.0
        # added_mass_min = 4.0
        
        # randomize_com_displacement = True
        # # com_displacement_range_xy = [-0.25, 0.25]
        # com_displacement_range_z = [-0.05, 0.05]

        # com_displacement_xy_min = 0.05
        # com_displacement_xy_max = 0.25
        

        
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
            dof_tau = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        rendered_envs_idx = [i for i in range(0, 1, 1)]  # number of environments to be rendered
        add_camera = True

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
        stiffness = {'joint': 10.0}   # [N*m/rad]
        damping   = {'joint': 0.25}     # [N*m*s/rad]
        
        action_scale = [0.25, 0.25, 0.25]    # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = [10.0, 10.0, 10.0] # action scale:  target torque = torque_scale * tau_action + defaultTorque
        
        
        dt =  0.01     # control frequency 200Hz
        decimation = 5  # decimation: Number of control action updates @ sim DT per policy DT

        # Assumed order - tau_ff, tau_fb
        # tradeoff_init_weights  = [0.80, 1.4]
        tradeoff_init_weights  = [0.50, 2.00]
        tradeoff_final_weights = [1.00, 1.00]
        tradeoff_steps = 10
        tradeoff_threshold = 0.60
        use_tradeoff_curriculum = False

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 1.00  # [rad] ~ 40 degrees
        pitch_threshold   = 0.7  # [rad] ~ 30 degrees
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
        only_positive_rewards = False

        use_reward_curriculum = True

        max_contact_force = 400.0
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination      = 0.0
            collision        = -1.0
            dof_pos_limits        = -5.0
            dof_close_to_default  = -0.5
            torque_limits         = -1.0
            
            no_motion_penalty     = 0.0
            alive_bonus           = 0.1

            stand_still_contact = -0.01
            stand_still         = -0.1

            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5
            dof_tracking      = 0.1
            aligned_torques   = 0.1
            sparse_contacts   = 0.01
            foot_swing  = 0.00
            
            # smoothness and stability
            lin_vel_z        = -2.0
            base_height      = -1.5
            ang_vel_xy       = -0.1
            orientation      = -2.0
            dof_acc          = -2.5e-7
            joint_power      = -2.e-5
            joint_power_dist = -1.e-5
            torques          = 0.0     # don't need to use this when we already have joint power above...

            # Zero out some values that are used in the individual reward classes below
            action_rate       = -0.001
            action_smoothness = -0.001

            feedforward_torques   = -2.0e-4
            feedback_torques      = -2.0e-4
            act_close_to_default    = -0.001
            dof_act_limits          = -1.0

            # promot stable WB locomotion
            # wb_dynamics = 0.1
            robust_wb_dynamics = 0.00
            # stable_grf_dynamics      = 0.0
            # floating_base_stability  = 0.01

            # gait
            feet_air_time    = 0.5            # tracking reward for long steps
            max_contact_time = 0.5            # penalty for feet being in contact for too long
            foot_clearance   = 0.5            # tracking reward for feet reaching the desired clearance
            foot_slip        = -0.1           # penalty for feet slipping
            feet_contact_forces = -2.0e-1     # penalty for high contact forces on the feet
            raibert  = 0.01                   # tracking reward foot placement in x/y-plane
            front_back_separation = -0.01     # penalty for small distance between front and back feet during contact

        class reward_curriculum():
            curr_reward_keys = ["action_rate", "action_smoothness", "feedback_torques", "feet_contact_forces",
                                "ang_vel_xy", "base_height", "lin_vel_z", "orientation", "act_close_to_default",
                                "front_back_separation", "feedforward_torques", "dof_act_limits"]
            
            curr_reward_bounds = {"action_rate":[-1.0e-3, -1.0e-2],
                                  "action_smoothness":[-1.0e-3, -1.0e-2],
                                  "feedback_torques":[-2.25e-4, -2.75e-4],
                                  "feedforward_torques":[-2.0e-4, -2.5e-4],
                                  "feet_contact_forces":[-1.0e-1,-5.0e-1],
                                  "ang_vel_xy":[-0.05, -0.1],
                                  "base_height":[-1.0,-2.0],
                                  "lin_vel_z":[-1.0,-2.0],
                                  "orientation":[-1.0,-2.0],
                                  "act_close_to_default":[-1.0e-4, -1.0e-3],
                                  "front_back_separation":[-1.0e-2, -1.0e-1],
                                  "dof_act_limits":[-1.0, -2.0]
                                 }

            curr_steps = 1000
            warmup_steps = 2500

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
    
    class env( LeggedRobotCfg.env ):
        num_envs = 10
        num_observations = 57
        num_privileged_obs = 67 # robot_state + other privilged info + terrain_heights (121)
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 5
        grf_dim = 12
        whole_body_dim = 18
        debug = False # if debugging, visualize contacts,
        debug_viz = False # draw debug visualizations
        use_liquid = True
    class liquid():
        liquid_type = "water"
        liquid_volume = 6.0  # liters
        liquid_tank = "default"

class GO1DynamicWaterCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "OnPolicyRunnerDynamicWater"
    
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
        cenet_dec_out_dim = 57      # next obs (57) + grf_dim (12)

        # Actor/critic
        actor_shared_dim = 512
        actor_branch_layers = [256,128]
        critic_layers = [256,128,64]
        
        # Shared
        dropout = 0.1

        pinn_loss_weight = 0.01
        pinn_warmup = 100
        pinn_init_steps = 0

        # pretrained_path = "/home/oyoungquist/Research/LearningWBIC/genesis_lr_dreamwaq/logs/rss_go1_dynamic_unimodel/Jan17_23-35-09_unimodel_grf_pinn_100hz_full_posboot_01_p2/model_200.pt"
        # pretrained_path = "../../rsl_rl/modules/pretrained_models/rl_pos/Jan17_17-39-51_unimodel_grf_01_100hz_tanh_pos/model_1000.pt"
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        # learning_rate = 1.0e-3 #
        learning_rate = 3.0e-4 #
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
        policy_class_name = 'ActorCritic_Dynamic'
        algorithm_class_name = 'PPODynamic'
        num_steps_per_env = 100 # per iteration
        max_iterations = 5000 # number of policy updates
        grf_dim = 12
        
        # debug_warmpinn_wb
        run_name = 'unimodel_grf_pinn_100hz_full_posboot_01_finetune_full_02'
        experiment_name = 'rss_go1_dynamic_unimodel_basic'
        save_interval = 100
        
        
        load_run = "Jan24_11-29-18_ablation_unimodel_basic_100hz_posboot_01_finetune_02"
        checkpoint = 5000
        resume = True
        exp_data_path = ""

        # Load parameters for first function policy
        # run_name = 'test_01'
        # experiment_name = 'go1_dynamic'
        # save_interval = 50
        # load_run = "Dec01_18-26-31_test_01"
        # checkpoint = 3000