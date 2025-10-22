from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles =  {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = ["foot"]
        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base", "hip", "thigh"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
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
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        self_collisions = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.30
        feet_height_target = 0.075
        tracking_sigma = 0.25
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.2
            lin_vel_z = -1.0
            base_height = -50.0
            action_rate = -0.005
            similar_to_default = -0.1

    class commands( LeggedRobotCfg.commands ):
        resampling_time = 4.0
        num_commands = 3
        class ranges:
            lin_vel_x = [-1.0, 2.5] # min max [m/s]
            lin_vel_y = [-0.8, 0.8]   # min max [m/s]
            ang_vel_yaw = [-0.9, 0.9]    # min max [rad/s]

class Go1RoughCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name="OnPolicyRunner"
    num_steps_per_env=24
    save_interval=100
    empirical_normalization=None
    seed=1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        clip_param=0.2
        desired_kl=0.01
        entropy_coef=0.01
        gamma=0.99
        lam=0.95
        learning_rate=0.001
        max_grad_norm=1.0
        num_learning_epochs=5
        num_mini_batches=4
        schedule="adaptive"
        use_clipped_value_loss=True
        value_loss_coef=1.0
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        checkpoint=-1
        experiment_name="go1_stdalone"
        load_run=-1
        log_interval=1
        max_iterations=1500
        num_steps_per_env = 24
        record_interval=-1
        resume=False
        resume_path=None
        run_name=""
    class policy(LeggedRobotCfgPPO.policy):
        activation="elu"
        actor_hidden_dims=[512, 256, 128]
        critic_hidden_dims=[512, 256, 128]
        init_noise_std=1.0
        class_name="ActorCritic"