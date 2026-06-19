from legged_gym.envs.base.legged_robot import *
from collections import deque

class LeggedRobotDreamwaq(LeggedRobot):
    
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
        
        # Estimator labels
        self.estimator_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,         # 3
            self.simulator.link_contact_states, # contact states of hips, thighs, calfs, feet and base (4+4+4+4+1)=17
            torch.clip(self.simulator.feet_pos[:, :, 2] -
            torch.mean(self.simulator.height_around_feet, dim=-1) -
            self.cfg.rewards.foot_height_offset, -1, 1.),  # 4
        ), dim=-1)
        
        # next state
        self.next_state_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions * self.cfg.control.action_scale,
        ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                   # 3
            self.obs_buf,                 # num_observations
        ), dim=-1)
        
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
    
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
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.explicit_labels_buf, \
            self.next_state_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self) -> None:
        """Check termination conditions and update reset buffer.
        
        Evaluates three termination conditions:
            1. Contact termination: Body contacts with termination bodies exceed threshold.
            2. Orientation termination: Projected gravity exceeds maximum allowed value.
            3. Timeout termination: Episode exceeds maximum episode length.
        
        Updates the following buffers:
            - fail_buf: Tracks consecutive failures for graceful termination.
            - time_out_buf: Indicates episodes that timed out (not actual failures).
            - reset_buf: Indicates environments needing reset.
        """
        # if the dim of link_contact_forces is 4, then it has history (IsaacLab). shape [N, T, B, 3] (N: num_envs, T: history length, B: number of links with contact sensors)
        if len(self.simulator.link_contact_forces.shape) == 4:
            self.terminated_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.termination_contact_indices, :], dim=-1), dim=1)[0]
            self.penalized_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.penalized_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_max_force_z = torch.max(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2], dim=1)[0]
        else:
            self.terminated_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            self.penalized_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1)
            self.feet_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :], dim=-1)
            self.feet_max_force_z = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2]
        
        self.gap_reset_buf = self._check_unrecoverable_gap()

        fail_buf = torch.any(self.terminated_bodies_force_norm > 10.0, dim=1)
        # print(f"contact termination: {fail_buf}")
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity}")
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # print(f"time out: {self.time_out_buf}")

        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        ) | self.gap_reset_buf

    def _check_unrecoverable_gap(self):
        if (
            not hasattr(self.cfg, "termination")
            or not getattr(self.cfg.termination, "reset_unrecoverable_gaps", False)
            or self.cfg.terrain.mesh_type not in ("heightfield", "trimesh")
            or not self.cfg.terrain.obtain_terrain_info_around_feet
        ):
            self.gap_fall_counter.zero_()
            return torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        support_height = self.simulator.env_origins[:, 2].unsqueeze(1)
        terrain_under_feet = self.simulator.height_around_feet[:, :, 4]
        deep_void = terrain_under_feet < (
            support_height
            - self.cfg.termination.gap_terrain_depth_threshold
        )
        fallen_feet = deep_void & (
            self.simulator.feet_pos[:, :, 2]
            < support_height - self.cfg.termination.gap_foot_drop_threshold
        )
        enough_fallen_feet = (
            fallen_feet.sum(dim=1)
            >= self.cfg.termination.gap_min_fallen_feet
        )
        base_fallen = deep_void.any(dim=1) & (
            self.simulator.base_pos[:, 2]
            < self.simulator.env_origins[:, 2]
            - self.cfg.termination.gap_base_drop_threshold
        )
        falling_into_gap = enough_fallen_feet | base_fallen

        self.gap_fall_counter = torch.where(
            falling_into_gap,
            self.gap_fall_counter + 1,
            torch.zeros_like(self.gap_fall_counter),
        )
        return (
            self.gap_fall_counter
            >= self.cfg.termination.gap_reset_steps
        )
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, explicit_labels, next_state, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history, explicit_labels, next_state
    
    def get_observations(self):
        return self.obs_buf, self.privileged_obs_buf, self.obs_history, self.explicit_labels_buf, self.next_state_buf
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_history_obs = self.cfg.env.num_history_obs
        self.num_latent_dims = self.cfg.env.num_latent_dims
        self.num_explicit_dims = self.cfg.env.num_explicit_dims
        self.num_decoder_output = self.cfg.env.num_decoder_output
        
    def _init_buffers(self):
        super()._init_buffers()
        # obs_history
        self.gap_fall_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_observations,
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
        # next state
        self.next_state_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_decoder_output),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        # explicit info labels
        self.explicit_labels_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_explicit_dims),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        self.gap_fall_counter[env_ids] = 0
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0