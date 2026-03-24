from legged_gym.envs.base.legged_robot_amp import LeggedRobotAMP
import torch
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float
from collections import deque
import numpy as np

class K1AMP(LeggedRobotAMP):
    
    def compute_observations(self):
        
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            # key_body_pos_relative_to_base.flatten(start_dim=1),
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
            self.simulator.dr_friction_values,        # 1
            self.simulator.dr_added_base_mass,        # 1
            self.simulator.dr_rand_push_vels[:, :2],  # 2
            self.simulator.dr_base_com_bias,          # 3
            self.simulator.dr_kp_scale,               # num_dofs
            self.simulator.dr_kd_scale                # num_dofs
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel, # 3
            obs_buf,                                               # num_obs
            domain_randomization_info,                             # 51
        ), dim=-1)
        
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        
        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
    def _init_buffers(self):
        super()._init_buffers()
        # obs_history
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # critic observation buffer
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
    
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
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        default = self.simulator.default_dof_pos
        # Shoulder_Pitch(2,6), Shoulder_Roll(3,7), Elbow_Yaw(5,9)
        dof_pos[:, [2,3,5,6,7,9]] = default[:, [2,3,5,6,7,9]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 6), self.device)
        # Hip_Pitch(10,16), Hip_Roll(11,17)
        dof_pos[:, [10,11,16,17]] = default[:, [10,11,16,17]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 4), self.device)
        # Knee_Pitch(13,19)
        dof_pos[:, [13,19]] = default[:, [13,19]] + torch_rand_float(-0.1, 0.3, (len(env_ids), 2), self.device)
        # Ankle_Pitch(14,20)
        dof_pos[:, [14,20]] = default[:, [14,20]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 2), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        return noise_vec
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.2  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_distance(self):
        '''reward for feet distance'''
        feet_xy_distance = torch.norm(
            self.simulator.feet_pos[:, 0, [0, 1]] - self.simulator.feet_pos[:, 1, [0, 1]], dim=-1)
        return torch.max(torch.zeros_like(feet_xy_distance),
                         self.cfg.rewards.foot_distance_threshold - feet_xy_distance)
    
    def _reward_foot_flat(self):
        """Encourage foot to be flat when contact with the ground
        """
        foot_contact = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_indices, :], dim=-1) > 1.0
        foot_quat = self.simulator.feet_quat
        # calculate world z axis in foot frame
        z_axis_world = torch.tensor([0., 0., 1.], device=self.device).repeat(foot_quat.shape[0], foot_quat.shape[1], 1)
        foot_z_axis = quat_rotate_inverse(foot_quat, z_axis_world)
        foot_tilt = torch.abs(foot_z_axis[:, :, 0]) + torch.abs(foot_z_axis[:, :, 1])  # x and y components
        rew_foot_flat = torch.exp(-foot_tilt / 0.1)
        return torch.sum(rew_foot_flat * foot_contact, dim=1)
    
    def _reward_hip_yaw_pos(self):
        """Encourage hip yaw to be close to default position
        """
        hip_yaw = self.simulator.dof_pos[:, [12, 18]]
        return torch.sum(torch.square(hip_yaw - self.simulator.default_dof_pos[:, [12, 18]]), dim=1)