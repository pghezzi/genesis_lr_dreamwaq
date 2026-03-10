from .legged_robot_config import *

class LeggedRobotTSDepthCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_camera_envs = 1 # number of envs with depth camera, starting from the first env
        num_observations = 45
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        single_critic_obs_len = num_observations
        num_critic_obs = c_frame_stack * single_critic_obs_len

    class normalization( LeggedRobotCfg.normalization):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            depth_image = 2.0
            height_measurements = 1.0
        clip_actions = 100.0
    
class LeggedRobotTSDepthCfgPPO(LeggedRobotCfgPPO):
    distillation = False # false -> teacher training, true -> student training
    runner_class_name = 'TSDepthRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        cnn_input_channel = LeggedRobotTSDepthCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [4, 8]
        cnn_strides = [1, 1]
        cnn_fc_layer_dims = [128]
        combination_mlp_dims = [128, 32]
        cnn_kernel_sizes = [5, 3]
        rnn_type = 'gru'
        rnn_hidden_size = 256
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTSDepth"
        algorithm_class_name = "PPO_TSDepth"
        teacher_model_path = "" # path to the teacher model checkpoint for distillation learning, necessary for student training