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
import argparse

def play(args):

    args.task = "go1_rl2ac_watereval"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    
    for i in range(2):
        env_cfg.viewer.pos[i] = env_cfg.viewer.pos[i] - env_cfg.terrain.plane_length / 4
        env_cfg.viewer.lookat[i] = env_cfg.viewer.lookat[i] - env_cfg.terrain.plane_length / 4    
    
    env_cfg.noise.add_noise = True
    # Disable some of the domain randomization (our payload will handle that now)
    env_cfg.domain_rand.randomize_com_displacement = False
    env_cfg.domain_rand.randomize_pd_gain = False           # Maybe keep this on?
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False


    env_cfg.asset.fix_base_link = False
    env_cfg.env.debug_viz = False
    
    if RECORD_FRAMES or FOLLOW_ROBOT:
        env_cfg.viewer.add_camera = True  # use a extra camera for moving
    
    # for MOVE_CAMERA
    if MOVE_CAMERA:
        camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    
    # for FOLLOW_ROBOT
    if FOLLOW_ROBOT:
        camera_lookat_follow = np.array(env_cfg.viewer.lookat)
        camera_deviation_follow = np.array([0., 3., -1.])
        camera_position_follow = camera_lookat_follow - camera_deviation_follow
    
    
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

    env_cfg.liquid.liquid_type = args.liquid_type
    env_cfg.liquid.liquid_volume = args.liquid_volume  # liters
    train_cfg.runner.exp_data_path = f"exp_data/rl2ac_trained_model/rl2ac_{int(args.liquid_volume)}L{args.liquid_type}_push_01.csv"
    # env_cfg.env.use_liquid = args.use_liquid
    env_cfg.env.use_liquid = False

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
    
    if RECORD_FRAMES:
        env.floating_camera.start_recording()
    
    start_time = time.perf_counter() # Record the start time

    for i in range(10*int(env.max_episode_length)):
    # for i in range(100):
        actions, q_ref = policy(obs.detach(), obs_hist.detach())
        obs, _, obs_hist, rews, dones, infos, grfs = env.step(actions.detach(), q_ref.detach())

        rewards.append(rews.cpu().numpy().tolist())
        total_grfs.append(grfs.cpu().numpy().tolist())

        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)
            env.floating_camera.render()
        
        if FOLLOW_ROBOT:
            # refresh where camera looks at(robot 0 base)
            camera_lookat_follow = env.base_pos[robot_index, :].cpu().numpy()
            # refresh camera's position
            camera_position_follow = camera_lookat_follow - camera_deviation_follow
            env.set_camera(camera_position_follow, camera_lookat_follow)
            env.floating_camera.render()

        # this accounts for the fact that the sim is running twicw as fast for this approach
        #     doing this prevents an imbalance in collected data.
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
                'tau_pd':env.first_loop_feedback.detach().cpu().numpy().tolist(),
                'tau_comp':env.adaptive_torques.detach().cpu().numpy().tolist(),
                'failure':list(map(int, env.get_failure_idx().detach().cpu().numpy().tolist()))
            }
        )


    # logger.save_log()

    end_time = time.perf_counter()   # Record the end time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    print("Mean Position Rewards - ", np.mean(rewards))
    print("Mean GRF-forces - ", np.mean(total_grfs))
    
    if RECORD_FRAMES:
        try:
            filename_mp4 = f"{train_cfg.runner.experiment_name}_{train_cfg.runner.load_run}.mp4"
        except:
            from datetime import datetime
            filename_mp4 = f"{datetime.now().timestamp()}"
        
        env.floating_camera.stop_recording(save_to_filename=filename_mp4, fps=30)
        print("Saved recording to " + filename_mp4)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False  # only record frames in extra camera view
    MOVE_CAMERA   = False
    FOLLOW_ROBOT  = True
    assert not (MOVE_CAMERA and FOLLOW_ROBOT), "Cannot move camera and follow robot at the same time"
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='go2')
    parser.add_argument('--headless',       action='store_true', default=False)  # enable visualization by default
    parser.add_argument('-c', '--cpu',      action='store_true', default=False)  # use cuda by default
    parser.add_argument('-B', '--num_envs', type=int, default=None)
    parser.add_argument('--max_iterations', type=int, default=None)
    parser.add_argument('--resume',         type=str, default=None)
    parser.add_argument('-o', '--offline',  action='store_true', default=False)
    parser.add_argument('-d', '--device',   type=str, default='cuda')

    parser.add_argument('--debug',          action='store_true', default=False)
    parser.add_argument('--ckpt',           type=int, default=1000)
    
    parser.add_argument('--use_liquid',    type=bool, default='True')
    parser.add_argument('--liquid_type',   type=str, default='water', choices=['water', 'oil', 'gas'])
    parser.add_argument('--liquid_volume', type=float, default=4.0)

    args = parser.parse_args()
    
    play(args)
