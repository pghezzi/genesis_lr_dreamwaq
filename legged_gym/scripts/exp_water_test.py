import genesis as gs
gs.init(backend=gs.gpu, logging_level='warning')
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import time

from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.exp_data_logger import ExpLogger

import numpy as np
import torch
import torch.nn.functional as F


def play(args):

    args.task = "go1_dynamic_watereval"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 30)
    # env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    
    # for i in range(2):
    #     env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
    #     env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4    
    
    env_cfg.noise.add_noise = True
    # Disable some of the domain randomization (our payload will handle that now)
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False           # Maybe keep this on?
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False


    env_cfg.asset.fix_base_link = False
    env_cfg.env.debug_viz = False
    env_cfg.viewer.add_camera = False  # use a extra camera for moving
    
    
    # initial state randomization
    env_cfg.init_state.yaw_angle_range = [0., 0.]
    # velocity range
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
    
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.resampling_time = 10.0

    # load policy
    train_cfg.runner.resume = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    
    policy = ppo_runner.get_inference_policy(device=env.device)

    _, _ = env.reset()
    
    obs, obs_hist, _ = env.get_observations()
    
    if type(obs) == tuple:
        obs = obs[0]
    

    env.num_iters += 1
    # env.step_push()

    logger = ExpLogger(train_cfg.runner.exp_data_path)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    rewards = []
    total_grfs  = []

    print("Max - self.feedforward_tau_weight: ", torch.max(env.feedforward_tau_weight).item())
    print("Min - self.feedforward_tau_weight: ", torch.min(env.feedforward_tau_weight).item())
    print("Max - self.feedback_tau_weight: ", torch.max(env.feedback_tau_weight).item())
    print("Min - self.feedback_tau_weight: ", torch.min(env.feedback_tau_weight).item())
    start_time = time.perf_counter() # Record the start time

    for i in range(2*int(env.max_episode_length)):
    # for i in range(1000):
        actions = policy(obs.detach(), obs_hist.detach())
        obs, _, obs_hist, rews, dones, infos, grfs = env.step(actions.detach())

        rewards.append(rews.cpu().numpy().tolist())
        total_grfs.append(grfs.cpu().numpy().tolist())
            # {
            #     'dof_pos_target': actions_scaled[robot_index, joint_index].item(), 
            #     'dof_pos': env.dof_pos[robot_index, joint_index].item(),
            #     'dof_vel': env.dof_vel[robot_index, joint_index].item(),
            #     'dof_tau_target': torques_scaled[robot_index, joint_index].item(),
            #     'dof_torque': env.torques[robot_index, joint_index].item(),
            #     'command_x': env.commands[robot_index, 0].item(),
            #     'command_y': env.commands[robot_index, 1].item(),
            #     'command_yaw': env.commands[robot_index, 2].item(),
            #     'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
            #     'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
            #     'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
            #     'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
            #     'contact_forces_z': env.link_contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
            # }

        logger.log_states(
            {
                'base_cmd':env.commands.detach().cpu().numpy().tolist(),
                'base_pose':env.base_pos.detach().cpu().numpy().tolist(),
                'base_rpy':env.base_euler.detach().cpu().numpy().tolist(),
                'dof_pose':env.dof_pos.detach().cpu().numpy().tolist(),
                'base_lin_vel':env.base_lin_vel.detach().cpu().numpy().tolist(),
                'base_ang_vel':env.base_ang_vel.detach().cpu().numpy().tolist(),
                'dof_vel':env.dof_vel.detach().cpu().numpy().tolist(),
                'proj_grav':env.projected_gravity.detach().cpu().numpy().tolist(),
                'feet_pos':env.feet_pos.detach().cpu().numpy().tolist(),
                'tau_act':env.dof_tau.detach().cpu().numpy().tolist(),
                'grf':env.grfs_buf.detach().cpu().numpy().tolist(),
                'q_des':env.get_scaled_pos_actions().detach().cpu().numpy().tolist(),
                'tau_ff':env.feedforward_torques.detach().cpu().numpy().tolist(),
                # 'tau_pd':env.feedback_torques_init.detach().cpu().numpy().tolist(),
                'tau_pd':env.first_loop_feedback.detach().cpu().numpy().tolist(),
                'failure':list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist()))
            }
        )


    logger.save_log()

    end_time = time.perf_counter()   # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    print("Mean Position Rewards - ", np.mean(rewards))
    print("Mean GRF-forces - ", np.mean(total_grfs))

    env.shutdown_asynic_pino_workers()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = False
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
    args = get_args()
    play(args)
