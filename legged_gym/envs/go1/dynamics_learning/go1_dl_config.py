from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO1DynamicCfg( LeggedRobotCfg ):
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 69
        num_privileged_obs = 89
        num_actions = 12
        env_spacing = 0.5
        num_obs_hist = 5
        grf_dim = 12
        whole_body_dim = 18

    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        friction = 1.0
        restitution = 0.
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.38] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        
        # Double check this.. use WBIC with a joint target
        default_joint_torques = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': 0.75,  # [nM]
            'FL_hip_joint': -0.75,  # [nM]
            'RR_hip_joint': -0.75,  # [nM]
            'RL_hip_joint': 0.75,  # [nM]

            'FL_thigh_joint': 0.5,   # [rad]
            'RL_thigh_joint': 0.5,   # [rad]
            'FR_thigh_joint': -0.5,   # [rad]
            'RR_thigh_joint': -0.5,   # [rad]

            'FL_calf_joint': 4.0,   # [rad]
            'RL_calf_joint': 4.0,   # [rad]
            'FR_calf_joint': 4.0,   # [rad]
            'RR_calf_joint': 4.0,   # [rad]
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
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = True
        joint_armature_range = [0.015, 0.025]  # [N*m*s/rad]
        randomize_joint_stiffness = True
        joint_stiffness_range = [0.01, 0.02]
        randomize_joint_damping = True
        joint_damping_range = [0.25, 0.3]

    class noise (LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            dof_tau = 2.0
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = action_scale * pose_action + defaultAngle
        torque_scale = 2.5 # action scale:  target torque = torque_scale * tau_action + defaultTorque
        dt =  0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT

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
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip"]
        links_to_keep = ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot']
        self_collisions = True
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        foot_clearance_target = 0.05 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            termination = -200.0
            dof_pos_limits = -10.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -0.5
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.5
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class GO1DynmaicCfgPPO( LeggedRobotCfgPPO ):
    seed = 10
    runner_class_name = "OnPolicyRunnerDynamic" # Teacher-Student Runner
    class policy( LeggedRobotCfgPPO.policy ):
        # Context encoder
        cenet_enc_layers=[256,128,64]
        cenet_enc_latent_dim = 29
        cenet_velo_dim = 3

        # Context Decoder
        cenet_dec_input_dim = 32
        cenet_dec_layers = [64,128,256,128,92]
        cenet_dec_out_dim = 81

        # Actor/critic
        actor_shared_dim = 512
        actor_branch_layers = [256,128,64]
        
        # Shared
        dropout = 0.1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic_Dynamic'
        algorithm_class_name = 'PPODynamic'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates
        grf_dim = 12
        
        run_name = 'test_01'
        experiment_name = 'go1_dynamic'
        save_interval = 100
        load_run = "Jul21_17-07-50_"
        checkpoint = -1
        max_iterations = 1000