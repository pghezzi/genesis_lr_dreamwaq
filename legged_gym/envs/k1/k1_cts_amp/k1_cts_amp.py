from legged_gym.envs.base.legged_robot_cts import LeggedRobotCTS
import torch
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float
from legged_gym.utils.motion_loader import AMPLoader
import numpy as np

class K1_CTS_AMP(LeggedRobotCTS):
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.critic_obs_buf, \
            self.rew_buf, self.reset_buf, self.extras, self.reset_env_ids, self.terminal_amp_states
        
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, critic_obs, _, _, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history, critic_obs
    
    def reset_idx(self, env_ids):
        self.reset_env_ids = env_ids
        self.terminal_amp_states = self.get_amp_observations()[env_ids]
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        self._resample_commands(env_ids)
        _ = np.random.random()
        if self.cfg.init_state.reference_state_initialization \
            and _ < self.cfg.init_state.reference_state_initialization_prob:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_from_reference_motion(env_ids, frames)
            self._reset_root_states_from_reference_motion(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def get_amp_observations(self):
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        # Use base_lin_vel_w, base_ang_vel_w, dof_pos, dof_vel, key_body_pos_relative_to_base in the observations
        return torch.cat((
            self.simulator.base_lin_vel,              # 3
             self.simulator.base_ang_vel,             # 3
            self.simulator.dof_pos,                   # num_dofs
            self.simulator.dof_vel,                   # num_dofs
            key_body_pos_relative_to_base.flatten(start_dim=1), # num_key_bodies * 3
        ), dim=-1)
    
    def compute_observations(self):
        
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
                
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
            self.simulator.dr_friction_values,        # 1
            self.simulator.dr_added_base_mass,        # 1
            self.simulator.dr_rand_push_vels[:, :2],  # 2
            self.simulator.dr_base_com_bias,          # 3
            self.simulator.dr_kp_scale,               # num_dofs
            self.simulator.dr_kd_scale,               # num_dofs
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel, # 3
            self.obs_buf,                                               # num_obs
            domain_randomization_info,                             # 51
            self.feet_air_time,                                    # 2
            self.simulator.feet_pos[:, :, 2],                      # 2
            key_body_pos_relative_to_base.flatten(start_dim=1),    # num_key_bodies * 3
        ), dim=-1)
        
        self.critic_obs_deque.append(single_critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # Privileged observation, for privileged encoder
        self.privileged_obs_buf = torch.cat((
            domain_randomization_info,
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.feet_air_time,                                    # 2
            self.simulator.feet_pos[:, :, 2],                      # 2
            key_body_pos_relative_to_base.flatten(start_dim=1),    # num_key_bodies * 3
        ), dim=-1)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.reset_env_ids = None
        self.terminal_amp_states = None
        if self.cfg.init_state.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, 
                                        device=self.device, 
                                        time_between_frames=self.dt,
                                        num_dof=self.num_actions,
                                        num_key_bodies=len(self.simulator.key_body_indices))
        
    
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
    
    def _reset_dofs_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the dof positions and velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_dof_pos = self.amp_loader.get_dof_pos_batch(ref_motions)
        ref_dof_vel = self.amp_loader.get_dof_vel_batch(ref_motions)
        self.simulator.reset_dofs(env_ids, ref_dof_pos, ref_dof_vel)
        
    def _reset_root_states_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the root positions, orientations, linear and angular velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_base_pos = self.amp_loader.get_base_pos_batch(ref_motions)
        ref_base_pos[:, 2] = self.simulator.base_init_pos[2]
        base_pos = ref_base_pos + self.simulator.env_origins[env_ids]
        ref_base_rot = self.amp_loader.get_base_rot_batch(ref_motions)
        ref_base_lin_vel = self.amp_loader.get_base_lin_vel_batch(ref_motions)
        ref_base_ang_vel = self.amp_loader.get_base_ang_vel_batch(ref_motions)
        self.simulator.reset_root_states(env_ids, base_pos, ref_base_rot, ref_base_lin_vel, ref_base_ang_vel)
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf)
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
        contact = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :], dim=-1) > 1.
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
    
    def _reward_hip_yaw_roll_pos(self):
        """Encourage hip yaw to be close to default position
        """
        hip_yaw = self.simulator.dof_pos[:, [11,12,17,18]]
        return torch.sum(torch.square(hip_yaw - self.simulator.default_dof_pos[:, [11,12,17,18]]), dim=1)
    
    def _reward_arm_pos(self):
        """Encourage arm joints to be close to default position
        """
        arm_joints = self.simulator.dof_pos[:, [2,3,4,5,6,7,8,9]]
        return torch.sum(torch.square(arm_joints - self.simulator.default_dof_pos[:, [2,3,4,5,6,7,8,9]]), dim=1)
    
    def _reward_head_pos(self):
        """Encourage head joints to be close to default position
        """
        head_joints = self.simulator.dof_pos[:, :2]
        return torch.sum(torch.square(head_joints - self.simulator.default_dof_pos[:, :2]), dim=1)
    
    def _reward_feet_slip(self):
        '''penalize foot slip when in contact with the ground'''
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        contacts = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2] > 0.1
        slip_penalty = torch.sum(foot_vel_xy_norm * contacts, dim=1)
        return slip_penalty