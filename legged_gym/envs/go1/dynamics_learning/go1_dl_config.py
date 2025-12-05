from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1DynamicCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 1000
        num_observations = 57
        num_privileged_obs = 82
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
        
    class init_state( LeggedRobotCfg.init_state ):
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
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 2.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
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
            dof_tau = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class viewer:
        ref_env = 0
        pos = [2, 2, 2]       # [m]
        lookat = [0., 0, 1.]  # [m]
        rendered_envs_idx = [i for i in range(0, 5, 1)]  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(200, 205, 1)])  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(500, 505, 1)])  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(750, 755, 1)])  # number of environments to be rendered
        rendered_envs_idx.extend([i for i in range(900, 905, 1)])  # number of environments to be rendered

        # rendered_envs_idx.extend([i for i in range(1500, 1505, 1)])
        # rendered_envs_idx.extend([i for i in range(1900, 1905, 1)])
        # rendered_envs_idx.extend([i for i in range(3500, 3505, 1)])
        # rendered_envs_idx.extend([i for i in range(4000, 4005, 1)])
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
        #     default values are taken from - https://arxiv.org/pdf/1909.06586
        stiffness = {'joint': 3.0}   # [N*m/rad]
        damping   = {'joint': 0.075}     # [N*m*s/rad]
        # stiffness = {'joint': 20.0}   # [N*m/rad]
        # damping   = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.6               # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = [25.0, 25.0, 35.0] # action scale:  target torque = torque_scale * tau_action + defaultTorque
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

        # Assumed order - tau_ff, tau_fb
        tradeoff_init_weights = [0.01, 10.0]
        tradeoff_final_weights = [1.0, 1.0]
        tradeoff_steps = 1000

    class termination:
        termination_terms = ["roll", "pitch", "height_min", "height_max"]
        roll_threshold    = 0.7  # [rad] ~ 40 degrees
        pitch_threshold   = 0.7  # [rad] ~ 20 degrees
        height_min = 0.10        # [m]
        height_max = 1.50        # [m]
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.90
        soft_torque_limit = 0.90
        base_height_target = 0.30
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        
        foot_clearance_target = 0.075 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = False

        reward_curriculum = True

        max_contact_force = 200.0
        class scales( LeggedRobotCfg.rewards.scales ):
            # General
            termination      = -200.0
            collision        = -1.0
            dof_pos_limits        = -10.0
            dof_close_to_default  = -0.1
            torque_limits         = -10.0
            stand_still           = -0.1
            no_motion_penalty     = -1.00
            alive_bonus           = 0.5


            # command tracking
            tracking_lin_vel  = 1.0
            tracking_ang_vel  = 0.5


            # smoothness and stability
            lin_vel_z        = -2.0
            base_height      = -2.0
            ang_vel_xy       = -0.5
            orientation      = -3.0
            dof_acc          = -2.5e-7
            joint_power      = -2.e-5
            joint_power_dist = -1.e-5
            torques          = 0.0 # don't need to use this when we already have joint power above...

            # Zero out some values that are used in the individual reward classes vbelow
            action_rate       = 0.0
            action_smoothness = 0.0

            # promot stable WB locomotion
            wb_dynamics = -1.0e-4

            # gait
            feet_air_time  = 1.0
            foot_clearance = 0.5
            foot_slip      = -1.0e-2
            feet_contact_forces = -1e-2

            # new....
            # raibert     = -1.0e-4
            raibert     = 0.0

        class pos_scales():
            pos_action_rate       = -0.001   # new
            pos_action_smoothness = -0.001   # new
            feedback_torques      = -2.e-4  # new

        class tau_scales():
            # task_alignment        = -1.e-3
            task_alignment        = 0.0
            tau_action_rate       = -0.001   # new
            tau_action_smoothness = -0.001   # new
            feedforward_torques   = -2.e-4  # new

        class reward_curriculum():
            curr_reward_keys = ["tau_action_rate", "tau_action_smoothness", "feedforward_torques",
                                "pos_action_rate", "pos_action_smoothness", "feedback_torques",
                                "dof_close_to_default", "dof_acc", "joint_power", "joint_power_dist",
                                "wb_dynamics"]
            
            curr_reward_bounds = {"tau_action_rate":[-1.0e-5, -1.0e-2],
                                  "tau_action_smoothness":[-1.0e-5, -1.0e-2],
                                  "pos_action_rate":[-1.0e-5, -1.0e-2],
                                  "pos_action_smoothness":[-1.0e-5, -1.0e-2],
                                  "feedforward_torques":[-2.0e-4, -2.0e-3],
                                  "feedback_torques":[-2.0e-4, -2.0e-3],
                                  "dof_close_to_default":[-1.0e-4, -0.1],
                                  "dof_acc":[-1.0e-9, -2.5e-7],
                                  "joint_power":[-2.0e-5, -2.0e-3],
                                  "joint_power_dist":[-1.0e-5, -1.0e-3],
                                  "wb_dynamics":[-1e-8, -1e-3]
                                  }

            curr_steps = 2000
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

class GO1DynmaicCfgPPO( LeggedRobotCfgPPO ):
    seed = 1
    runner_class_name = "OnPolicyRunnerDynamic" # Teacher-Student Runner
    
    class policy( LeggedRobotCfgPPO.policy ):
        activation = 'swish' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid, swish (SiLU)
        init_noise_std = 1.0
        
        # Context encoder
        cenet_enc_layers=[128,64]
        cenet_enc_latent_dim = 16
        cenet_velo_dim = 3

        # Context Decoder
        cenet_dec_input_dim = 19
        cenet_dec_layers = [64,128]
        cenet_dec_out_dim = 57        # next obs (57) + grf_dim (12)

        # Actor/critic
        actor_shared_dim = 512
        actor_branch_layers = [256,128]
        critic_layers = [512,256,128,64]
        
        # Shared
        dropout = 0.1

        pinn_loss_weight = 1e-5

        # pretrained_path = "/home/oyoungquist/Research/LearningWBIC/genesis_lr_dreamwaq/rsl_rl/modules/pretrained_models/dynamic/11_22_202511_53_49_no_pinn/no_pinn_epoch_199.pth"

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.0e-4 #
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam   = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_Dynamic'
        algorithm_class_name = 'PPODynamic'
        num_steps_per_env = 24 # per iteration
        max_iterations = 5000 # number of policy updates
        grf_dim = 12
        
        run_name = 'debug_pinn_wb'
        experiment_name = 'go1_dynamic'
        save_interval = 50
        load_run = "Dec02_18-42-41_raibert_wbdyn"
        checkpoint = 2000

        # Load parameters for first function policy
        # run_name = 'test_01'
        # experiment_name = 'go1_dynamic'
        # save_interval = 50
        # load_run = "Dec01_18-26-31_test_01"
        # checkpoint = 3000
