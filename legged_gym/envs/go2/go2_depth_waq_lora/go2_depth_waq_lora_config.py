from legged_gym import *
from legged_gym.envs.base.legged_robot_dreamwaq_config import LeggedRobotDreamwaqCfg, LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg
from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq_config import Go2DepthWaqCfg, Go2DepthWaqCfgPPO, terrain_name, finetune

#assume everything is exactly the same with configs
class Go2DepthWaqLoraCfg( Go2DepthWaqCfg ):
    pass
    
class Go2DepthWaqLoraCfgPPO( Go2DepthWaqCfgPPO ):
    runner_class_name = "DreamWaQDepthLoraRunner"
    class policy( Go2DepthWaqCfgPPO.policy ):
        base_model = finetune
    class runner( Go2DepthWaqCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQDepthLora"
        experiment_name = f"go2_depth_waq_lora{'_' + terrain_name}"
        pre_trained = None