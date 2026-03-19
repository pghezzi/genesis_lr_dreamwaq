import numpy as np

from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot_ts_depth import LeggedRobotTSDepth
from legged_gym.utils.math_utils import torch_rand_float

class Go2TSDepth(LeggedRobotTSDepth):
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    (self.simulator._friction_values - 
                    self.friction_value_offset),            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    (self.simulator._kp_scale - 
                     self.kp_scale_offset),                 # num_actions
                    (self.simulator._kd_scale - 
                     self.kd_scale_offset)                  # num_actions
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 31
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,     # 3
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            height_relative_to_feet = (self.simulator.feet_pos[:, :, 2].unsqueeze(-1) - 
                                       self.simulator.height_around_feet).clip(-1, 1.).flatten(1, 2) # 9*number of feet=36
            critic_obs = torch.cat((critic_obs, 
                                    height_relative_to_feet,   # 36
                                    self.simulator.normal_vector_around_feet # 12
                                    ), dim=-1)
        if self.cfg.terrain.measure_heights: # 144
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.3 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec
        
        # Privileged observation, for privileged encoder
        self.privileged_obs_buf = torch.cat(
            (
                domain_randomization_info,                       # 32
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,     # 3
            ),
            dim=-1,
        )
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            self.privileged_obs_buf = torch.cat((
                                    self.privileged_obs_buf, 
                                    height_relative_to_feet,   # 36
                                    self.simulator.normal_vector_around_feet # 12
                                    ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            self.privileged_obs_buf = torch.cat(
                (
                    self.privileged_obs_buf,                   # previous
                    self.simulator.link_contact_states,        # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 144
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
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
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        heading_error = self.heading - self.commands[:, 3]
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    