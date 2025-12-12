import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from collections import deque
import torch.nn.functional as F
import pinocchio as pn


class LeggedRobotGo1Dynamic(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.height_samples = None
        self.debug_viz = self.cfg.env.debug_viz
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_device, headless)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def get_observations(self):
        return self.obs_buf, self.obs_history, self.base_lin_vel
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, 2*self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # clip the predicted actions
        
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        
        self.actions = F.tanh(self.actions)

        # Perform random control delay if approperiate
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions = self.action_queue[torch.arange(
                self.num_envs), self.action_delay].clone()
        
        # run simulation steps with current action and PD feedback control
        for _ in range(self.cfg.control.decimation):  # use self-implemented pd controller
            self.torques = self._compute_torques(self.actions)
        
            if self.num_build_envs == 0:
                torques = self.torques.squeeze()
                self.robot.control_dofs_force(torques, self.motors_dof_idx)
            else:
                self.robot.control_dofs_force(
                    self.torques, self.motors_dof_idx)
        
            self.scene.step()
        
            self.dof_pos[:] = self.robot.get_dofs_position(
                self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(
                self.motors_dof_idx)
        
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
            
        # total_episodic_rewards = self.rew_buf+self.pos_rew_buf+self.tau_rew_buf
        # rew_cv = torch.std(total_episodic_rewards) / torch.mean(total_episodic_rewards)
        # pboot_rew = 1.0 - torch.abs(torch.tanh(rew_cv))
        # print("rew_cv: ", rew_cv)
        # print("pboot_rew: ", pboot_rew)
        # print()
        #         
        # Retunring some extra stuff and two separate reward functions
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, (self.rew_buf+self.pos_rew_buf), (self.rew_buf + self.tau_rew_buf), self.reset_buf, self.extras, (self.grfs_buf * self.obs_scales.grf)

    def get_prev_obs(self):
        return self.last_obs_buf, self.last_obs_hist, self.llast_obs_buf, self.llast_obs_hist
    
    def get_pinn_wb_dynamics(self):
        #           total GT forces  ,  generalized mass mat, bias vector
        return self.contact_forces_buff, self.wb_mass_mat_buff, self.wb_bias_vec_buff, self.torso_6dof_acceleration

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        
        base_quat_rel = gs_quat_mul(self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1)))
        
        self.base_euler = gs_quat2euler(base_quat_rel)
        inv_base_quat = inv_quat(self.base_quat)
        
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)  # trasform to base frame
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.link_contact_forces[:] = self.robot.get_links_net_contact_force()
        self.feet_pos[:] = self.robot.get_links_pos()[:, self.feet_indices, :]
        self.feet_vel[:] = self.robot.get_links_vel()[:, self.feet_indices, :]
        self.dof_tau[:] = self.robot.get_dofs_force(self.motors_dof_idx)

        # Used to train the model, so reoganize this to match the model inidices
        #     extract the values used to calculate the dynamics consitentcy reward separately.
        self.grfs_buf[:] = self.robot.get_links_net_contact_force()[:, self.feet_indices, :].reshape(self.base_pos.shape[0], self.grf_dim)

        # All the below is done in the pinocchio indexing scheme [FL, FR, RL, RR]
        # Use the Pinocchio library to calculate the (1) contact forces and (2) whole-body dynamics of the robot for use
        #     in the dynamic consistency reward. All done in WORLD FRAME!
        
        #     extract the contact forces in pinocchio order
        contact_temp = self.robot.get_links_net_contact_force()[:, self.pino_feet_indices, :]
        wb_dynamics_list = []
        wb_contact_forces_list = []
        pinn_mass_mats = []
        pinn_bias_vecs = []
        torso_accelerations = []

        # indexing scheme used to return the wb_value back into the model's indexing scheme
        correct_pino_2_model_wb_idxs = [0,1,2,3,4,5]
        correct_pino_2_model_wb_idxs.extend(self.pino_2_model_joint_act_map)

        base_velo_world = self.robot.get_vel()
        base_ang_velo_world = self.robot.get_ang()

        for i in range(self.dof_pos.shape[0]):
            # print(self.dof_pos[0])
            pino_dof_pos = self.dof_pos[i,self.model_2_pino_joint_map].cpu().numpy().tolist()
            pino_dof_vel = self.dof_vel[i,self.model_2_pino_joint_map].cpu().numpy().tolist()
            # print(pino_dof_pos)
            # Used for approximating the accelerations...
            pino_prev_dof_velo = self.last_dof_vel[i,self.model_2_pino_joint_map].cpu().numpy().tolist()
            
            # construct the whole body pose
            pino_wb_pose = []
            pino_wb_pose.extend(self.base_pos[i].cpu().numpy().tolist())
            # Genesis is w,x,y,z quat, Pinocchio wants x,y,z,w    https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_2d-practical-exercises_23-invkine.html
            temp_quat = self.base_quat[i].cpu().numpy()[1:].tolist()
            temp_quat.append(float(self.base_quat[i].cpu().numpy()[0]))
            pino_wb_pose.extend(temp_quat)
            pino_wb_pose.extend(pino_dof_pos)
            pino_wb_pose = np.array(pino_wb_pose)

            # construct the whole body velocity
            pino_wb_velo = []
            pino_wb_velo.extend(base_velo_world[i].cpu().numpy().tolist())
            pino_wb_velo.extend(base_ang_velo_world[i].cpu().numpy().tolist())
            pino_wb_velo.extend(pino_dof_vel)
            pino_wb_velo = np.array(pino_wb_velo)

            # construct the previous whole body velocity (used to approximate accelerations)
            pino_prev_wb_velo = []
            pino_prev_wb_velo.extend(self.last_base_world_lin_vel[i].cpu().numpy().tolist())
            pino_prev_wb_velo.extend(self.last_base_world_ang_vel[i].cpu().numpy().tolist())
            pino_prev_wb_velo.extend(pino_prev_dof_velo)
            pino_prev_wb_velo = np.array(pino_prev_wb_velo)

            # now use a simple backwards finite-difference for acceleration approximation
            pino_wb_acc = (pino_wb_velo - pino_prev_wb_velo) / self.dt

            # necessary for PINN updates
            torso_accelerations.append(torch.from_numpy(pino_wb_acc[0:6]))

            # Calculate the generalized mass matrix and bias forces
            aq0 = np.zeros(self.pino_model.nv)
            #     compute dynamic drift -- Coriolis, centrifugal, gravity
            b = pn.rnea(self.pino_model, self.pino_data, pino_wb_pose, pino_wb_velo, aq0)   # batch_size x 18
            #     compute mass matrix M
            M = pn.crba(self.pino_model, self.pino_data, pino_wb_pose)   # batch_size, (18x18)

            # use the calculated values to approximate the whole-body dynamics
            wb_dynamics = np.squeeze(np.matmul(M,pino_wb_acc) + b)

            # reshape and append to the batch-list
            wb_dynamics_list.append(torch.from_numpy(wb_dynamics[correct_pino_2_model_wb_idxs]))

            # Log the dyanmics values for use in the external PINN loss
            reshaped_M = M[correct_pino_2_model_wb_idxs,:]
            reshaped_M = reshaped_M[:,correct_pino_2_model_wb_idxs]
            reshaped_b = b[correct_pino_2_model_wb_idxs]
            pinn_mass_mats.append(torch.from_numpy(reshaped_M))
            pinn_bias_vecs.append(torch.from_numpy(reshaped_b))

            # Now calculate the contact forces impact on the dynamics
            pino_jacobains = []
            for i, foot_name in enumerate(self.pino_foot_names):
                # print(foot_name)
                foot_frame_id = self.pino_model.getFrameId(foot_name)
                pino_jacobains.append(pn.computeFrameJacobian(self.pino_model, self.pino_data, pino_wb_pose, foot_frame_id, pn.ReferenceFrame.LOCAL_WORLD_ALIGNED)[0:3,:])

            full_jacobian = np.concatenate(pino_jacobains, axis=0) # 12x18

            reshaped_contacts = contact_temp.reshape(contact_temp.shape[0], 12).unsqueeze(2)[i].cpu().numpy()  # 12x1

            contact_forces = np.squeeze(np.matmul(full_jacobian.transpose(), reshaped_contacts)) # 18

            wb_contact_forces_list.append(torch.from_numpy(contact_forces[correct_pino_2_model_wb_idxs]))
        # end pinocchio loop

        # now stack the tensor lists to get the necessary state values
        self.wb_dynamics_buff[:]    = torch.stack(wb_dynamics_list).to(self.device)               # batch x 18
        self.contact_forces_buff[:] = torch.stack(wb_contact_forces_list).to(self.device)      # batch x 18

        # print(self.wb_dynamics_buff.shape)
        # print(self.contact_forces_buff.shape)

        self.wb_mass_mat_buff[:] = torch.stack(pinn_mass_mats).to(self.device)
        self.wb_bias_vec_buff[:] = torch.stack(pinn_bias_vecs).to(self.device)
        self.torso_6dof_acceleration[:] = torch.stack(torso_accelerations).to(self.device)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_base_pos_out_of_bound()
        self.check_termination()
       
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        
        if self.num_build_envs > 0:
            self.reset_idx(env_ids)
        
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_base_world_lin_vel[:] = base_velo_world[:]
        self.last_base_world_ang_vel[:] = base_ang_velo_world[:]

        if self.debug_viz:
            self._draw_debug_vis()

    def check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        x_out_of_bound = (self.base_pos[:, 0] >= self.terrain_x_range[1]) | (
            self.base_pos[:, 0] <= self.terrain_x_range[0])
        y_out_of_bound = (self.base_pos[:, 1] >= self.terrain_y_range[1]) | (
            self.base_pos[:, 1] <= self.terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        envs_idx = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        # reset base position to initial position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.link_contact_forces[:, self.termination_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        
        if hasattr(self.cfg, "termination"):
            # more sophisticated termination conditions
            rpy = gs_quat2euler(self.base_quat)
            r, p = rpy[:,0], rpy[:,1]
            r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
            p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
            height = self.base_pos[:, 2] - self.env_origins[:, 2]
            
            if "roll" in self.cfg.termination.termination_terms:
                r_term_buff = torch.abs(r) > self.cfg.termination.roll_threshold
                self.reset_buf |= r_term_buff
            if "pitch" in self.cfg.termination.termination_terms:
                p_term_buff = torch.abs(p) > self.cfg.termination.pitch_threshold
                self.reset_buf |= p_term_buff
            if "height_min" in self.cfg.termination.termination_terms:
                height_term_buff = height < self.cfg.termination.height_min
                self.reset_buf |= height_term_buff
            if "height_max" in self.cfg.termination.termination_terms:
                height_term_buff = height > self.cfg.termination.height_max
                self.reset_buf |= height_term_buff
        
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # domain randomization
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(env_ids)
        
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(env_ids)
        
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(env_ids)
        
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        
        if self.cfg.domain_rand.randomize_joint_stiffness:
            self._randomize_joint_stiffness(env_ids)
        
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        
        if self.cfg.domain_rand.randomize_pd_gain:
            self._randomize_joint_pd(env_ids)

        # 2 things - (1) randomly sample paramaters for water tank using helper function
        #          - (1) helper function will control the curriculum of the "range" of randomness

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.

        self.last_base_world_lin_vel[env_ids] = 0.
        self.last_base_world_ang_vel[env_ids] = 0.

        self.feet_air_time[env_ids] = 0.
        self.feet_air_time_raibert[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.grfs_buf[env_ids] = 0.
        self.contact_forces_buff[env_ids] = 0.
        self.wb_dynamics_buff[env_ids] = 0.
        # clear obs history for the envs that are reset
        self.last_obs_buf[env_ids] = 0.
        self.llast_obs_buf[env_ids] = 0.
        
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0

        # PINN stuff
        self.wb_mass_mat_buff[env_ids]  = 0.
        self.wb_bias_vec_buff[env_ids]  = 0.
        self.last_obs_hist[env_ids]     = 0. 
        self.llast_obs_hist[env_ids]     = 0. 
        self.torso_6dof_acceleration[env_ids] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # reset action queue and delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue[env_ids] *= 0.
            self.action_queue[env_ids] = 0.
            self.action_delay[env_ids] = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self.device, requires_grad=False)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # Accumulate the shared general rewards
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            # print("Shared reward - ", name)
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        
        # Accumulate position control specific rewards
        self.pos_rew_buf[:] = 0.
        for i in range(len(self.pos_reward_functions)):
            name = self.pos_reward_names[i]
            # print("Position reward - ", name)
            rew = self.pos_reward_functions[i]() * self.pos_reward_scales[name]
            self.pos_rew_buf += rew
            self.episode_sums[name] += rew
        
        # Accumulate torque control specific rewards
        self.tau_rew_buf[:] = 0.
        for i in range(len(self.tau_reward_functions)):
            name = self.tau_reward_names[i]
            # print("Torque reward - ", name)
            rew = self.tau_reward_functions[i]() * self.tau_reward_scales[name]
            self.tau_rew_buf += rew
            self.episode_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        self.llast_obs_buf = self.last_obs_buf.clone().detach()
        self.last_obs_buf = self.obs_buf.clone().detach()
        self.obs_buf = torch.cat((self.commands[:,:3]*self.commands_scale,     # 3 DOF
                                  self.projected_gravity,                      # 3 DOF
                                  self.base_ang_vel * self.obs_scales.ang_vel, # 3 DOF
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12 DOF
                                  self.dof_vel * self.obs_scales.dof_vel,      # 12 DOF       
                                #   (self.dof_tau - self.default_dof_tau)*self.obs_scales.dof_tau,    # 12 DOF
                                  self.actions[:,0:12],                        # 12 DOF
                                  self.actions[:,12:24]), dim=-1)              # 12 DOF  total of - 57
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - \
                             1) * self.noise_scale_vec
            
        # push last_obs_buf to obs_history
        self.llast_obs_hist = self.last_obs_hist.clone().detach()
        self.last_obs_hist = self.obs_history.clone().detach()
        
        self.obs_history_deque.append(self.last_obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )

        if self.cfg.domain_rand.randomize_ctrl_delay:
            # normalize to [0, 1]
            ctrl_delay = (self.action_delay /
                          self.cfg.domain_rand.ctrl_delay_step_range[1]).unsqueeze(1)

        if self.num_privileged_obs is not None:
            # TODO for liquid payloads -> added liquid mass + vscosity values to priv. obs.

            # Clip the GRF values to help stablize the critic...

            self.privileged_obs_buf = torch.cat(
                (   
                    self.obs_buf,                 # 57 DOF
                    self.base_lin_vel * self.obs_scales.lin_vel,  # 3 DOF
                    self.grfs_buf * self.obs_scales.grf,          # 12 DOF
                    self._friction_values,        # 1
                    self._added_base_mass,        # 1
                    self._base_com_bias,          # 3
                    self._rand_push_vels[:, :2],  # 2
                    self._joint_armature,         # 1
                    self._joint_stiffness,        # 1
                    self._joint_damping,          # 1
                    # mass of water tank
                    # stickness of water in tank
                ),
                dim=-1,
            )   # 3 total of 82 
            
            # print(self.privileged_obs_buf)

    # payload
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.sim_dt,
                substeps=self.sim_substeps),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.cfg.control.decimation),
                camera_pos=np.array(self.cfg.viewer.pos),
                camera_lookat=np.array(self.cfg.viewer.lookat),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx= self.cfg.viewer.rendered_envs_idx),
            rigid_options=gs.options.RigidOptions(
                dt=self.sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=self.cfg.asset.self_collisions,
                batch_dofs_info=True,   # batch dof info for all envs
                batch_joints_info=True,
                batch_links_info=True,
            ),
            show_viewer=not self.headless,
        )
        # query rigid solver
        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            elif isinstance(solver, AvatarSolver):
                continue
            self.rigid_solver = solver

        # add camera if needed
        if self.cfg.viewer.add_camera:
            self._setup_camera()

        # add terrain
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type =='plane':
            self.terrain = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        elif mesh_type =='heightfield':
            self.utils_terrain = Terrain(self.cfg.terrain)
            self._create_heightfield()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        self.terrain.set_friction(self.cfg.terrain.friction)
        # specify the boundary of the heightfield
        self.terrain_x_range = torch.zeros(2, device=self.device)
        self.terrain_y_range = torch.zeros(2, device=self.device)
        
        if self.cfg.terrain.mesh_type =='heightfield':
            self.terrain_x_range[0] = -self.cfg.terrain.border_size + 1.0  # give a small margin(1.0m)
            self.terrain_x_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_rows * self.cfg.terrain.terrain_length - 1.0
            self.terrain_y_range[0] = -self.cfg.terrain.border_size + 1.0
            self.terrain_y_range[1] = self.cfg.terrain.border_size + \
                self.cfg.terrain.num_cols * self.cfg.terrain.terrain_width - 1.0
        
        elif self.cfg.terrain.mesh_type =='plane': # the plane used has limited size, 
                                                 # and the origin of the world is at the center of the plane
            self.terrain_x_range[0] = -self.cfg.terrain.plane_length/2+1
            self.terrain_x_range[1] = self.cfg.terrain.plane_length/2-1
            self.terrain_y_range[0] = -self.cfg.terrain.plane_length/2+1  # the plane is a square
            self.terrain_y_range[1] = self.cfg.terrain.plane_length/2-1
        
        self._create_envs()

    def set_camera(self, pos, lookat):
        """ Set camera position and direction
        """
        self.floating_camera.set_pose(
            pos=pos,
            lookat=lookat
        )

    # ------------- Callbacks --------------
    def _setup_camera(self):
        ''' Set camera position and direction
        '''
        self.floating_camera = self.scene.add_camera(
            res= (1280, 960),
            pos=np.array(self.cfg.viewer.pos),
            lookat=np.array(self.cfg.viewer.lookat),
            fov=40,
            GUI=True,
        )

        self._recording = False
        self._recorded_frames = []

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 2] - heading), -1.0, 1.0)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = gs_rand_float(
            *self.cfg.commands.ranges.lin_vel_x, (len(env_ids),), self.device)
        self.commands[env_ids, 1] = gs_rand_float(
            *self.cfg.commands.ranges.lin_vel_y, (len(env_ids),), self.device)
        self.commands[env_ids, 2] = gs_rand_float(
            *self.cfg.commands.ranges.ang_vel_yaw, (len(env_ids),), self.device)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(
            self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        
        # randomly zero out the various elements of the commands
        

    def _compute_torques(self, actions):
        # control_type = 'P'
        # Pull out the position control actions
        pos_actions = actions[:,0:12]
        # pull out the torque control actions
        tau_actions = actions[:,12:24]
        # Scale the position actions

        repeat_pos_scales = torch.from_numpy(np.array(self.cfg.control.action_scale)).repeat(1,4).to(self.device)
        # actions_scaled = pos_actions * self.cfg.control.action_scale
        actions_scaled = pos_actions * repeat_pos_scales

        # Calculate the feedback-control torques
        #     include PD scaling values 
        self.feedback_torques = (
            (self._kp_scale * self.p_gains) * (actions_scaled + self.default_dof_pos - self.dof_pos) - (self._kd_scale * self.d_gains) * self.dof_vel
        )
        # Combine with the scaled + offset torque actions
        # print("FeedForward Torque - ")
        # print((tau_actions * self.cfg.control.torque_scale + self.default_dof_tau)[0:4,:])
        # print("Feedback Torques - ")
        # print(feedback_torques[0:4,:])
        
        repeat_torque_scales = torch.from_numpy(np.array(self.cfg.control.torque_scale)).repeat(1,4).to(self.device)
        
        self.feedforward_torques = (tau_actions * repeat_torque_scales + self.default_dof_tau)
        
        torques = (self.feedforward_tau_weight) * self.feedforward_torques + (self.feedback_tau_weight)*self.feedback_torques

        # self.feedforward_torques *= self.feedforward_tau_weight
        # self.feedback_torques *= self.feedback_tau_weight

        # torques = self.feedback_torques
        
        # torques = self.feedforward_torques + self.feedback_torques
        # print(self.feedforward_torques[0:5,:])
        # print(self.feedforward_tau_weight * self.feedforward_torques[0:5,:])
        # print("self.default_dof_tau")
        # print(self.default_dof_tau)
        # print("self.feedforward_torques")
        # print(self.feedforward_torques[0:5,:])
        # print("self.feedback_torques")
        # print(self.feedback_tau_weight * self.feedback_torques[0:5,:])
        # print("---------------------------------------")
        # print("Output Torques")
        # print(torques[0:5,:])
        # Have the limit be exceeded a little bit to get reward feedback based on exceeding the limits
        # return torch.clip(torques, -self.torque_limits, self.torque_limits)
        return torques
        # return self.feedback_torques

    def _get_pinn_actions(self, actions):
        # apply the tanh activation to scale between [-1, 1]
        actions = F.tanh(actions)
        # Pull out the position control actions
        pos_actions = actions[:,0:12]
        # pull out the torque control actions
        tau_actions = actions[:,12:24]
        
        # Scale and shift the position actions
        repeat_pos_scales = torch.from_numpy(np.array(self.cfg.control.action_scale)).repeat(1,4).to(self.device)
        # actions_scaled = pos_actions * self.cfg.control.action_scale
        actions_scaled = pos_actions * repeat_pos_scales
        target_dof_pos = actions_scaled + self.default_dof_pos
        
        # Scale and shift the torque actions
        repeat_torque_scales = torch.from_numpy(np.array(self.cfg.control.torque_scale)).repeat(1,4).to(self.device)
        feedforward_torques = (tau_actions * repeat_torque_scales + self.default_dof_tau)

        return target_dof_pos, feedforward_torques


    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.cfg.control.action_scale
        target_dof_pos = actions_scaled + self.default_dof_pos

        return target_dof_pos

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[envs_idx] = (self.default_dof_pos) + gs_rand_float(-0.3, 0.3, (len(envs_idx), self.num_actions), self.device)

        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # # randomly sample inital torques close to the default values
        # self.dof_tau[envs_idx] = (self.default_dof_tau) 
        # # + gs_rand_float(-1.0, 1.0, (len(envs_idx), self.num_actions), self.device)
        
        # # set the control torques approperiately
        # self.robot.control_dofs_force(self.dof_tau[envs_idx], self.motors_dof_idx, envs_idx)
        
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _reset_root_states(self, envs_idx):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base pos: xy [-1, 1]
        if self.custom_origins:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
            self.base_pos[envs_idx, :2] += gs_rand_float(-1.0, 1.0, (len(envs_idx), 2), self.device)
        else:
            self.base_pos[envs_idx] = self.base_init_pos
            self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        # base quat
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)  # roll, pitch [-0.1, 0.1]
        base_euler[:, 2] = gs_rand_float(*self.cfg.init_state.yaw_angle_range, (len(envs_idx),), self.device)  # yaw angle
        self.base_quat[envs_idx] = gs_quat_mul(
            gs_euler2quat(base_euler), self.base_quat[envs_idx],)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(
            self.global_gravity, inv_base_quat)

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = (
            gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))
        self.base_ang_vel[envs_idx] = (
            gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device))
        base_vel = torch.concat(
            [self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[
                                     0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        if self.push_interval_s > 0 and not self.debug:
            max_push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
            # in Genesis, base link also has DOF, it's 6DOF if not fixed.
            dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy,
                                     max_push_vel_xy, (self.num_envs, 2), self.device)
            self._rand_push_vels[:, :2] = push_vel.detach().clone()
            push_vel[((self.common_step_counter + self.env_identities) %
                      int(self.push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.utils_terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(
            self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >=self.max_terrain_level,
                                                   torch.randint_like(
                                                       self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],
            self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            

            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        
        self.add_noise = self.cfg.noise.add_noise
        
        noise_scales = self.cfg.noise.noise_scales
        
        noise_level = self.cfg.noise.noise_level
        
        # input commands
        noise_vec[:3] = 0.
        # projected gravity vector
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # angular velocity
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        # leg joint positions
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        # leg joint velocities
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        # # leg joint torques
        # noise_vec[33:45] = noise_scales.dof_tau * \
        #     noise_level * self.obs_scales.dof_tau
        
        # # mightttt add noise to these, but will already be fairly noise from RL, thinking about it
        # # previous joint position actions
        # noise_vec[45:57] = 0.
        # # previous joint torque actions
        # noise_vec[57:69] = 0.
        
        # mightttt add noise to these, but will already be fairly noise from RL, thinking about it
        # previous joint position actions
        noise_vec[33:45] = 0.
        # previous joint torque actions
        noise_vec[45:57] = 0.
        
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        self.forward_vec[:, 0] = 1.0
        
        self.base_init_pos = torch.tensor(
            self.cfg.init_state.pos, device=self.device
        )
        
        self.base_init_quat = torch.tensor(
            self.cfg.init_state.rot, device=self.device
        )
        
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        self.last_base_world_lin_vel = torch.zeros_like(self.base_lin_vel)

        self.last_base_world_ang_vel = torch.zeros_like(self.base_ang_vel)

        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=gs.tc_float)
        
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device,
            dtype=gs.tc_float,
                                           requires_grad=False,)
        self.actions = torch.zeros(
            (self.num_envs, 2*self.num_actions), device=self.device, dtype=gs.tc_float)
        
        self.last_actions = torch.zeros_like(self.actions)
        
        self.llast_actions = torch.zeros(self.num_envs, 2*self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        
        self.dof_pos = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        
        self.dof_vel = torch.zeros_like(self.dof_pos)
        
        self.last_dof_vel = torch.zeros_like(self.dof_pos)
        
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float)
        
        self.feet_air_time_raibert = torch.zeros(
            (self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_float)
        
        self.last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int)

        self.raibert_last_contacts = torch.zeros((self.num_envs, len(self.feet_indices)), device=self.device, dtype=gs.tc_int) 
        
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )
        
        self.feet_pos = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float
        )
        
        self.feet_vel = torch.zeros(
            (self.num_envs, len(self.feet_indices), 3), device=self.device, dtype=gs.tc_float
        )
        
        self.continuous_push = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=gs.tc_float
        )
        
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        
        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )
        
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        
        # obs_history
        self.last_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_observations),
            dtype=gs.tc_float,
            device=self.device,
        )

        self.llast_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_observations),
            dtype=gs.tc_float,
            device=self.device,
        )
        
        self.obs_history_deque = deque(maxlen=self.cfg.env.num_obs_hist)

        self.obs_history = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=gs.tc_float)
        
        for _ in range(self.cfg.env.num_obs_hist):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_observations,
                    dtype=gs.tc_float,
                    device=self.device,
                )
            )

        self.grfs_buf = torch.zeros(
            (self.num_envs, self.grf_dim), device=self.device, dtype=gs.tc_float)
        
        self.contact_forces_buff = torch.zeros(
            (self.num_envs, self.wb_dim), device=self.device, dtype=gs.tc_float)
        
        self.wb_dynamics_buff = torch.zeros(
            (self.num_envs, self.wb_dim), device=self.device, dtype=gs.tc_float)
        
        # Holds the generalized mass matrix computed by pinocchio, reshaped to match the model order (FR, FL, RR, RL)
        self.wb_mass_mat_buff = torch.zeros(
            (self.num_envs, self.wb_dim, self.wb_dim), device=self.device, dtype=gs.tc_float)
        
        # Hold the bias vector (gravity, corilis, centerfugal) calculated by pinocchio, reshaped to match the model order
        self.wb_bias_vec_buff = torch.zeros(
            (self.num_envs, self.wb_dim), device=self.device, dtype=gs.tc_float)
        
        self.last_obs_hist = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=gs.tc_float)
        
        self.llast_obs_hist = torch.zeros(
            (self.num_envs, self.num_obs * self.num_obs_hist), device=self.device, dtype=gs.tc_float)
        
        self.torso_6dof_acceleration = torch.zeros(self.num_envs, 6, device=self.device, dtype=gs.tc_float)
        
        self.dof_tau = torch.zeros_like(self.dof_pos)

        self.pos_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.tau_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        # randomize action delay
        if self.cfg.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(
                self.num_envs, self.cfg.domain_rand.ctrl_delay_step_range[1]+1, 2*self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay = torch.randint(self.cfg.domain_rand.ctrl_delay_step_range[0],
                                              self.cfg.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        self.default_dof_pos = torch.tensor(
            [self.cfg.init_state.default_joint_angles[name]
                for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )

        self.default_dof_tau = torch.tensor(
            [self.cfg.init_state.default_joint_torques[name]
                for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=gs.tc_float,
        )
        # PD control
        stiffness = self.cfg.control.stiffness
        damping = self.cfg.control.damping

        self.p_gains, self.d_gains = [], []
        for dof_name in self.cfg.asset.dof_names:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)
        # PD control params
        self.robot.set_dofs_kp(self.p_gains, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.d_gains, self.motors_dof_idx)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
            Splits into three reward groups (1) position control rewards (2) torque control rewards and (3) rewards shared between the two tasks
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale ==0:
                self.reward_scales.pop(key)
            else:
                # print("Non-zero shared reward + scale - ", key)
                # print(self.reward_scales[key])
                self.reward_scales[key] *= self.dt

        for key in list(self.pos_reward_scales.keys()):
            scale = self.pos_reward_scales[key]
            if scale ==0:
                self.pos_reward_scales.pop(key)
            else:
                # print("Non-zero position reward + scale - ", key)
                # print(self.pos_reward_scales[key])
                self.pos_reward_scales[key] *= self.dt

        for key in list(self.tau_reward_scales.keys()):
            scale = self.tau_reward_scales[key]
            if scale ==0:
                self.tau_reward_scales.pop(key)
            else:
                # print("Non-zero torque reward + scale - ", key)
                # print(self.tau_reward_scales[key])
                self.tau_reward_scales[key] *= self.dt
        
        # prepare list of functions
        # These are the general rewards....
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # position control rewards
        self.pos_reward_functions = []
        self.pos_reward_names = []
        for name, scale in self.pos_reward_scales.items():
            if name =="termination":
                continue
            
            self.pos_reward_names.append(name)
            name = '_reward_' + name
            self.pos_reward_functions.append(getattr(self, name))
        
        # torque control rewards
        self.tau_reward_functions = []
        self.tau_reward_names = []
        for name, scale in self.tau_reward_scales.items():
            if name =="termination":
                continue
            
            self.tau_reward_names.append(name)
            name = '_reward_' + name
            self.tau_reward_functions.append(getattr(self, name))

        # print( (self.reward_scales.keys() | self.pos_reward_scales.keys() | self.tau_reward_scales.keys()))

        if self.use_reward_curriculum:
            self.step_reward_curriculum()

        # reward episode sums, across all reward groups
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=gs.tc_float, device=self.device, requires_grad=False)
                             for name in (self.reward_scales.keys() | self.pos_reward_scales.keys() | self.tau_reward_scales.keys())}

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        self.terrain = self.scene.add_entity(
            gs.morphs.Terrain(
                pos=(-self.cfg.terrain.border_size, - \
                     self.cfg.terrain.border_size, 0.0),
                horizontal_scale=self.cfg.terrain.horizontal_scale,
                vertical_scale=self.cfg.terrain.vertical_scale,
                height_field=self.utils_terrain.height_field_raw,
            )
        )
        self.height_samples = torch.tensor(self.utils_terrain.heightsamples).view(
            self.utils_terrain.tot_rows, self.utils_terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, create entity
             2. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links= True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep= self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed= self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        # add water tanks to robots....

        # build
        self.scene.build(n_envs=self.num_envs)

        self._get_env_origins()

        self._init_domain_params()

        # name to indices
        self.motors_dof_idx = [self.robot.get_joint(
            name).dof_start for name in self.cfg.asset.dof_names]


        # find link indices, termination links, penalized links, and feet, utility function
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        ###
        #  Load an istance of the robot model within a pinocchio rigid body dynamics library class
        #      and create the necessary index maps.
        ###
        
        # Create a pinocchio dynamics model and data container
        self.pino_model = pn.buildModelFromUrdf(os.path.join(asset_root, asset_file), pn.JointModelFreeFlyer())
        self.pino_data  = self.pino_model.createData()

        # Create the joint mappings from model-2-pino and pino-2-model - model: [FR, FL, RR, RL], pino: [FL, FR, RL, RR]
        pino_dof_names = [name for name in self.pino_model.names[2:]]   # skip the universe and base joints

        # Maps from the [FR, FL, RR, RL] leg order used by the model to the [FL, FR, RL, RR] order used by pinocchio 
        #       I have confirmed that this is the order the joints load in for these URDF's but this should
        #       be safe for aribitrary orderings.
        self.model_2_pino_joint_map = []
        for dof_name in pino_dof_names:
            self.model_2_pino_joint_map.append(self.dof_names.index(dof_name))

        print("self.model_2_pino_joint_map")
        print(self.model_2_pino_joint_map)
        
        # Maps from pinocchio's leg order to the [FR, FL, RR, RL] ordering used by the learning model and
        #      enforced in this code. Note, pinocchio's DOF positions uses a quat for the orientation
        #      and so has a lightly different indexing scheme from the output of the elements of the 
        #      dynamics equations we will use, so we need both
        self.pino_2_model_joint_pos_map = []
        self.pino_2_model_joint_act_map = []
        for joint_name in self.dof_names:
            joint_id = self.pino_model.getJointId(joint_name)  # pull out the pinocchio idx for this joint
            joint = self.pino_model.joints[joint_id]           # pull out the joint itself
            v_idx = joint.idx_v                                # Get the start index of the joint's DoFs in the velocity vector (v)
            q_idx = joint.idx_q                                # Get the start index of the joint's DoFs in the configuration vector (q)
            # The joints we care about only have a single DOF...
            self.pino_2_model_joint_act_map.append(v_idx)
            self.pino_2_model_joint_pos_map.append(q_idx)

        print("self.pino_2_model_joint_pos_map and self.pino_2_model_joint_act_map")
        print(self.pino_2_model_joint_pos_map)
        print(self.pino_2_model_joint_act_map)


        # Also need a separate list of foot names and indicies... so stupid...
        self.pino_foot_names = []

        for frame in self.pino_model.frames:
            name = frame.name
            if name.endswith("foot"):
                self.pino_foot_names.append(name)

        self.pino_feet_indices = find_link_indices(self.pino_foot_names)

        self.termination_indices = find_link_indices(
            self.cfg.asset.terminate_after_contacts_on)
        
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names}")
        print("termination link indices:", self.termination_indices)
        
        self.penalized_indices = find_link_indices(
            self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_indices}")
       
        self.feet_names = [
            link.name for link in self.robot.links if self.cfg.asset.foot_name[0] in link.name]
        self.feet_indices = find_link_indices(self.feet_names)
        print(f"feet link indices: {self.feet_indices}")
       
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0

        # dof position limits
        self.dof_pos_limits = torch.stack(
            self.robot.get_dofs_limit(self.motors_dof_idx), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motors_dof_idx)[1]

        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )

        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
        # randomize joint armature
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self.num_envs))
        # randomize joint stiffness
        if self.cfg.domain_rand.randomize_joint_stiffness:
            self._randomize_joint_stiffness(np.arange(self.num_envs))
        # randomize joint damping
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self.num_envs))

    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_stiffness = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Water-tank simulation random params.... 
        
        self._kp_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=gs.tc_float, device=self.device)
        self._kd_scale = torch.ones(
            self.num_envs, self.num_actions, dtype=gs.tc_float, device=self.device)
        

    def _randomize_joint_pd(self, env_ids=None):
        
        self._kp_scale[env_ids] = gs_rand_float(
            self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        
        self._kd_scale[env_ids] = gs_rand_float(
            self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)


    def _randomize_friction(self, env_ids=None):
        ''' Randomize friction of all links'''
        min_friction, max_friction = self.cfg.domain_rand.friction_range

        solver = self.rigid_solver

        ratios = gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) \
        * (max_friction - min_friction) + min_friction
        
        self._friction_values[env_ids] = ratios[:,
            0].unsqueeze(1).detach().clone()

        solver.set_geoms_friction_ratio(
            ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids=None):
        ''' Randomize base mass'''
        min_mass, max_mass = self.cfg.domain_rand.added_mass_range
        base_link_id = 1
        added_mass = gs.rand((len(env_ids), 1), dtype=float) * \
                             (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].detach().clone()
        self.rigid_solver.set_links_mass_shift(
            added_mass, [base_link_id, ], env_ids)

    def _randomize_com_displacement(self, env_ids):

        min_displacement, max_displacement = self.cfg.domain_rand.com_displacement_range
        base_link_id = 1

        com_displacement = gs.rand((len(env_ids), 1, 3), dtype=float) \
        * (max_displacement - min_displacement) + min_displacement
        self._base_com_bias[env_ids] = com_displacement[:,
            0, :].detach().clone()

        self.rigid_solver.set_links_COM_shift(
            com_displacement, [base_link_id,], env_ids)

    def _randomize_joint_armature(self, env_ids):
        """ Randomize joint armature of the robot
        """
        min_armature, max_armature = self.cfg.domain_rand.joint_armature_range
        armature = torch.rand((len(env_ids), 1), dtype=gs.tc_float, device=self.device) \
        * (max_armature - min_armature) + min_armature
        self._joint_armature[env_ids, 0] = armature[:, 0].detach().clone()
        armature = armature.repeat(1, self.num_actions)  # repeat for all motors
        self.robot.set_dofs_armature(
            armature, self.motors_dof_idx, envs_idx=env_ids)

    def _randomize_joint_stiffness(self, env_ids):
        """ Randomize joint stiffness of the robot
        """
        min_stiffness, max_stiffness = self.cfg.domain_rand.joint_stiffness_range
        stiffness = torch.rand((len(env_ids), 1), dtype=gs.tc_float, device=self.device) \
        * (max_stiffness - min_stiffness) + min_stiffness
        self._joint_stiffness[env_ids, 0] = stiffness[:, 0].detach().clone()
        stiffness = stiffness.repeat(1, self.num_actions)
        self.robot.set_dofs_stiffness(
            stiffness, self.motors_dof_idx, envs_idx=env_ids)

    def _randomize_joint_damping(self, env_ids):
        """ Randomize joint damping of the robot
        """
        min_damping, max_damping = self.cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids), 1), dtype=gs.tc_float, device=self.device) \
        * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids, 0] = damping[:, 0].detach().clone()
        damping = damping.repeat(1, self.num_actions)
        self.robot.set_dofs_damping(
            damping, self.motors_dof_idx, envs_idx=env_ids)
        
    def step_tradeoff_curriculum(self):
        self.feedforward_tau_weight = self.tradeoff_upperbounds[0]
        self.feedback_tau_weight = self.tradeoff_upperbounds[1]

        if self.num_iters < self.tradeoff_num_steps:
            raw_step = float(self.num_iters)/float(self.tradeoff_num_steps)   # between [0,1]
            print(raw_step)
            remapped_step = 12.0 * raw_step + (-6.0)  # between [-6, 6]
            print(remapped_step)
            gentle_step = 1.0 / (1.0 + np.exp(-remapped_step))
            print(gentle_step)

            self.feedforward_tau_weight = gentle_step*self.bound_diff[0] + self.tradeoff_lowerbounds[0]
            self.feedback_tau_weight    = gentle_step*self.bound_diff[1] + self.tradeoff_lowerbounds[1]

        print("self.feedforward_tau_weight: ", self.feedforward_tau_weight)
        print("self.feedback_tau_weight: ", self.feedback_tau_weight)

    def step_reward_curriculum(self):
        # Safety catch
        if not self.use_reward_curriculum:
            return
        
        # initialize the policy with fixed-lower bound
        if self.num_iters < self.reward_warmup_steps:
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = self.reward_curr_bounds[key][0] * self.dt
                if key in self.pos_reward_scales.keys():
                    self.pos_reward_scales[key] = self.reward_curr_bounds[key][0] * self.dt
                if key in self.tau_reward_scales:
                    self.tau_reward_scales[key] = self.reward_curr_bounds[key][0] * self.dt
        # Gradually increase the regularization strength
        elif self.num_iters > self.reward_warmup_steps and (self.num_iters - self.reward_warmup_steps) < self.reward_curr_steps:
            print("Stepping Reward Curriculum")
            adjusted_iter = self.num_iters - self.reward_warmup_steps
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = ((float(adjusted_iter)/float(self.reward_curr_steps))*self.reward_bound_diffs[key] + self.reward_curr_bounds[key][0])*self.dt
                if key in self.pos_reward_scales.keys():
                    self.pos_reward_scales[key] = ((float(adjusted_iter)/float(self.reward_curr_steps))*self.reward_bound_diffs[key] + self.reward_curr_bounds[key][0])*self.dt
                if key in self.tau_reward_scales:
                    self.tau_reward_scales[key] = ((float(adjusted_iter)/float(self.reward_curr_steps))*self.reward_bound_diffs[key] + self.reward_curr_bounds[key][0])*self.dt
        # Fix the regularization strength to the upper-bound
        else:
            # by default set the reward to the upper bound
            for key in self.reward_curr_keys:
                if key in self.reward_scales.keys():
                    self.reward_scales[key] = self.reward_curr_bounds[key][1] * self.dt
                if key in self.pos_reward_scales.keys():
                    self.pos_reward_scales[key] = self.reward_curr_bounds[key][1] * self.dt
                if key in self.tau_reward_scales:
                    self.tau_reward_scales[key] = self.reward_curr_bounds[key][1] * self.dt

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.dt
        # use self-implemented pd controller
        self.sim_dt = self.dt / self.cfg.control.decimation
        self.sim_substeps = 1
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        
        self.pos_reward_scales = class_to_dict(self.cfg.rewards.pos_scales)
        self.tau_reward_scales = class_to_dict(self.cfg.rewards.tau_scales)

        self.use_reward_curriculum = self.cfg.rewards.reward_curriculum

        self.reward_curr_keys = self.cfg.rewards.reward_curriculum.curr_reward_keys
        self.reward_curr_bounds = self.cfg.rewards.reward_curriculum.curr_reward_bounds
        self.reward_curr_steps = self.cfg.rewards.reward_curriculum.curr_steps
        self.reward_warmup_steps = self.cfg.rewards.reward_curriculum.warmup_steps

        self.reward_bound_diffs = {}
        for key in self.reward_curr_keys:
            self.reward_bound_diffs[key] = self.reward_curr_bounds[key][1] - self.reward_curr_bounds[key][0]

        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        
        if self.cfg.terrain.mesh_type not in ['heightfield']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.push_interval_s = self.cfg.domain_rand.push_interval_s

        # determine privileged observation offset to normalize privileged observations
        self.friction_value_offset = (self.cfg.domain_rand.friction_range[0] + 
                                      self.cfg.domain_rand.friction_range[1]) / 2  # mean value
        self.kp_scale_offset = (self.cfg.domain_rand.kp_range[0] +
                                self.cfg.domain_rand.kp_range[1]) / 2  # mean value
        self.kd_scale_offset = (self.cfg.domain_rand.kd_range[0] +
                                self.cfg.domain_rand.kd_range[1]) / 2  # mean value

        self.wb_dim = self.cfg.env.whole_body_dim
        self.grf_dim = self.cfg.env.grf_dim

        self.dof_names = self.cfg.asset.dof_names
        self.debug = self.cfg.env.debug

        self.num_obs_hist = self.cfg.env.num_obs_hist

        # Tradeoff curriculum stuff...
        self.tradeoff_lowerbounds = np.array(self.cfg.control.tradeoff_init_weights)
        self.tradeoff_upperbounds = np.array(self.cfg.control.tradeoff_final_weights)
        self.tradeoff_num_steps = self.cfg.control.tradeoff_steps
        self.bound_diff = self.tradeoff_upperbounds - self.tradeoff_lowerbounds 
        self.num_iters = 0
        self.feedforward_tau_weight = 1.0
        self.feedback_tau_weight = 1.0

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height points
        if not self.cfg.terrain.measure_heights:
            return
        self.scene.clear_debug_objects()
        height_points = quat_apply_yaw(self.base_quat.repeat(
            1, self.num_height_points), self.height_points)
        height_points[0, :, 0] += self.base_pos[0, 0]
        height_points[0, :, 1] += self.base_pos[0, 1]
        height_points[0, :, 2] = self.measured_heights[0, :]
        # print(f"shape of height_points: ", height_points.shape) # (num_envs, num_points, 3)
        self.scene.draw_debug_spheres(height_points[0, :], radius=0.03, color=(0, 0, 1, 0.7))  # only draw for the first env

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (
                self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.utils_terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,
                self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self.cfg.env.env_spacing
            if num_rows * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2 or \
                num_cols * self.cfg.env.env_spacing > self.cfg.terrain.plane_length / 2:
                spacing = min((self.cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self.cfg.terrain.plane_length / 2) / (num_cols-1))
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            self.env_origins[:, 0] -= self.cfg.terrain.plane_length / 4
            self.env_origins[:, 1] -= self.cfg.terrain.plane_length / 4

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y,
                         device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x,
                         device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points,
                             3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(
                1, self.num_height_points), self.height_points[env_ids]) + (self.base_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(
                1, self.num_height_points), self.height_points) + (self.base_pos[:, :3]).unsqueeze(1)

        points += self.cfg.terrain.border_size
        points = (points/self.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.cfg.terrain.vertical_scale

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        tol = 0.02
        base_height = torch.mean(self.base_pos[:, 2].unsqueeze(
            1) - self.measured_heights, dim=1)
        
        height_diffs = base_height - self.cfg.rewards.base_height_target

        # within_tol = torch.abs(height_diffs) < tol

        # height_rewards = torch.where(within_tol, torch.ones_like(height_diffs)*0.1, -torch.abs(height_diffs))

        rew = torch.square(height_diffs)
        
        return rew

    def _reward_torques(self):
        # Penalize the FeedForward torques
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_feedback_torques(self):
        return torch.sum(torch.square(self.feedback_torques),dim=1)
    
    def _reward_feedforward_torques(self):
        return torch.sum(torch.square(self.feedforward_torques),dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_pos_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,0:12] - self.actions[:,0:12]), dim=1)
    
    def _reward_tau_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions[:,12:24] - self.actions[:,12:24]), dim=1)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost
    
    def _reward_pos_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,0:12] - 2*self.last_actions[:,0:12] + self.llast_actions[:,0:12]), dim=-1)
        return action_smoothness_cost
    
    def _reward_tau_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(
            self.actions[:,12:24] - 2*self.last_actions[:,12:24] + self.llast_actions[:,12:24]), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.link_contact_forces[:, self.penalized_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - \
                          self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        return torch.exp(-4.0*lin_vel_error)

    def _reward_joint_power(self):
        # penalize large amounts of motor power
        return torch.sum(torch.abs(self.dof_vel * self.torques), dim=1)

    def _reward_joint_power_dist(self):
        # Penalize uneven distributions of motor power
        return torch.var(self.torques*self.dof_vel, dim=1)

    def _reward_foot_slip(self):
        # penalize feet that are in-contact for any movement in the x/y direction
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        return  torch.sum(torch.square(contact * torch.sum(self.feet_vel[:,:,:2], dim=-1)), dim=-1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        # return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        return torch.exp(-4.0*ang_vel_error)
    
    def _reward_task_alignment(self):
        # Penalize un-aligned torques (feedforward torque is in the opposite direction as feedback)
        un_algined = torch.sum((self.feedforward_torques * self.feedback_torques < 0), dim=-1) * -1.0
        # Reward algined torques 
        algined = torch.sum((self.feedforward_torques * self.feedback_torques > 0), dim=-1)
        return un_algined + algined    
    
    # Consider tests when this objective is filtered by in-contact legs
    def _reward_wb_dynamics(self):
        # reward the combined torque + position control values that result in stable next-step whole-body dynamics
        #     this also "guides" the policy to select complimentary position and torque values
        # augment the torques vector to include 6 zeros for the unactuated torso DOF's
        # wb_torques = torch.concatenate((torch.zeros((self.torques.shape[0], 6), device=self.device, dtype=gs.tc_float), self.torques), dim=1)
        error = self.wb_dynamics_buff[:,6:] - self.contact_forces_buff[:,6:] - self.torques
        
        # filter this error signal by feet that are in-contact, otherwise this just penalizes the torque magnitude of the swing-legs, which we are
        #    already doing with other reward signals
        contact_filter = self.contact_forces_buff[:,6:] > 0.0
        filtered_error = contact_filter * error
        
        return torch.exp(-torch.norm(filtered_error, dim=1))    

    # Rewards control torques (feedforward + feedback) and blanace with the GRF profile at the joint-level
    #     thereby encouraging control torques and contact forces that conform to the systems rigid-body dynamics 
    def _reward_stable_grf_dynamics(self):  
        error = self.torques - self.contact_forces_buff[:,6:] 

        # filter this error signal by feet that are in-contact, otherwise this just penalizes the torque magnitude of the swing-legs, which we are
        #    already doing with other reward signals
        contact_filter = self.contact_forces_buff[:,6:] > 0.0
        filtered_error = contact_filter * error

        return torch.exp(-torch.norm(filtered_error, dim=1))
    
    # adjust this to eventually include the forces applied by a payload during training (in the negative z-direction)
    def _reward_floating_base_stability(self):
        # Calculate the actual mass of the system
        adjusted_base_mass = self.robot.get_links_inertial_mass()[:,0] + self._added_base_mass.squeeze(-1)

        # Compute the GRF induced accelerations
        temp_grfs = torch.sum(self.robot.get_links_net_contact_force()[:, self.feet_indices, :], dim=1)   # (num_envs, 3)
        grf_acc = temp_grfs / adjusted_base_mass.unsqueeze(-1)

        # calculate the error between the observed COM movement (accelerations) and the GRF profile
        torso_acc_error = torch.norm(self.torso_6dof_acceleration[:,0:3] - grf_acc, dim=-1)

        return torch.exp(-torso_acc_error) 


    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.link_contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_no_motion_penalty(self):
        cmd_mag = torch.norm(self.commands[:, :2], dim=1)
        should_move = cmd_mag > 0.1

        vel_mag = torch.norm(self.base_lin_vel[:, :2], dim=1)

        # negative distance from movement threshold
        lack_of_motion = (0.1 - vel_mag).clamp(min=0.0)

        penalty = lack_of_motion * should_move.float()
        
        return penalty

    def _reward_foot_contact(self):
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact = contact * 0.25
        return torch.sum(contact, dim=1)
    
    def _reward_alive_bonus(self):
        return ~self.reset_buf
    
    def _reward_dof_close_to_default(self):
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        """
        foot_vel_xy_norm = torch.norm(self.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.feet_pos[:, :, 2] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
        # return clearance_error

    # The definition and calculation of this reward is inspired by the footstep selection calculations in https://arxiv.org/pdf/1909.06586
    def _reward_raibert(self):
        # Some constants. Will optimize later...
        # Assume a decent "walk" (~1m/s) stance time of 0.5 seconds
        stance_time = 0.5
        width_offset = 0.06
        raibert_gain = 0.03
        # The arrays below assume the foot ordering of FR, FL, RR, RL
        side_signs = torch.from_numpy(np.array([-1,1,-1,1])).float().to(self.device)
        hip_offsets = torch.from_numpy(np.array([[0.19, -0.047, 0.0], [0.19, 0.047, 0.0], [-0.19, -0.047, 0.0], [-0.19, 0.047, 0.0]])).float().to(self.device)
        
        # contact filtering...
        contact = self.link_contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.raibert_last_contacts)
        self.raibert_last_contacts = contact
        first_contact = (self.feet_air_time_raibert > 0.) * contact_filt
        self.feet_air_time_raibert += self.dt
        self.feet_air_time_raibert *= ~contact_filt

        inv_base_quat = inv_quat(self.base_quat)

        # calculate the Raibert Hueristic footstep location
        # Perform for each foot in order or FR, FL, RR, RL
        #     eventually want (num_env, 4, 2) - raibert is only concerned with x/y position
        raibert_foot_pos = []
        for i in range(len(self.feet_indices)):
            # Calculate Raibert huersitic foot contact locations in x/y-plane
            offset = torch.zeros((self.feet_pos.shape[0], 3),dtype=gs.tc_float).to(self.device)     # (num_env, 3)
            offset[:,1] = side_signs[i] * width_offset                                              # (num_env, 3)
            probot_frame = hip_offsets[i] + offset                                                  # (num_env, 3)
            z_rot_mats = self._build_raibert_rew_rot_mat(-self.commands[:,2] * stance_time * 0.5)   # (num_env, 3, 3)
            corrected_probot_frame = torch.bmm(z_rot_mats, probot_frame.unsqueeze(-1)).squeeze(-1)  # (num_env, 3)
            
            # This is the result of eq. 13
            basic_foot_pose = self.base_pos + transform_by_quat(corrected_probot_frame, inv_base_quat)  # (num_env, 3)

            # now calculate the more complicated Raibert hueristic
            #     symmetry hueristic
            raibert_xy = self.base_lin_vel[:,:2] * (0.5 * stance_time) + raibert_gain * (self.base_lin_vel[:,:2] - self.commands[:,:2])  # (num_env, 2)
            #     centrifugal hueristic
            raibert_xy[:,0] += 0.5 * (self.base_pos[:,2]/9.81) *   self.base_lin_vel[:,1] * self.commands[:,2]   # x-axis 
            raibert_xy[:,1] += 0.5 * (self.base_pos[:,2]/9.81) * (-self.base_lin_vel[:,0] * self.commands[:,2])  # y-axis

            # now we can calculate the final heursitic foot placement on the ground(xy)-plane in the base-frame 
            heuristic_foot_pos = torch.zeros((self.num_envs, 3), dtype=gs.tc_float)                    # (num_env, 3)
            heuristic_foot_pos[:,0] = basic_foot_pose[:,0] + raibert_xy[:,0]
            heuristic_foot_pos[:,1] = basic_foot_pose[:,1] + raibert_xy[:,1]
            
            # Append the calculations for this foot
            raibert_foot_pos.append(heuristic_foot_pos)

        # Now we have a list like (4, num_env, 3), we want (num_env, 4, 3)
        raibert_foot_pos = torch.stack(raibert_foot_pos).to(self.device)   # (4, num_env, 3)
        raibert_foot_pos = raibert_foot_pos.permute(1,0,2)                 # (num_env, 4, 3)

        # Calculate the error, ignore height, that is covered elsewhere
        foot_error_xy = torch.sum(self.feet_pos[:,:,:2] - raibert_foot_pos[:,:,:2], dim=-1)  # (num_env, 4)

        # filter by the first contact AND square the error
        raibert_error = torch.norm(first_contact * foot_error_xy, dim=-1)

        return torch.exp(-raibert_error / self.cfg.rewards.foot_clearance_tracking_sigma)
        # return torch.exp(-raibert_error)
        # return raibert_error

    # A rotation about the z-axis
    def _build_raibert_rew_rot_mat(self, theta):
        s = torch.sin(theta)        # (num_env,)
        c = torch.cos(theta)        # (num_env,)

        batch_rot_mats = torch.zeros((self.num_envs, 3, 3), dtype=gs.tc_float).to(self.device)

        # Fill in the non-zero entries
        batch_rot_mats[:,0,0] = c[:]
        batch_rot_mats[:,0,1] = s[:]
        batch_rot_mats[:,1,0] = -s[:]
        batch_rot_mats[:,1,1] = c[:]
        batch_rot_mats[:,2,2] = 1.0

        return batch_rot_mats
