from .legged_robot_config import *
from legged_gym import LEGGED_GYM_ROOT_DIR
import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/*")

class LeggedRobotAMPCfg(LeggedRobotCfg):
    
    class env(LeggedRobotCfg.env):
        amp_motion_files = MOTION_FILES
    
    class init_state(LeggedRobotCfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7
        

class LeggedRobotAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'AMPRunner'
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000

    class runner( LeggedRobotCfgPPO.runner ):
        algorithm_class_name = 'PPO_AMP'
        
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_discr_hidden_dims = [1024, 512]