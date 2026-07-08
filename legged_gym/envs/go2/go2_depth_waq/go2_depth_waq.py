import torch
import random
import numpy as np
from legged_gym.envs.base.legged_robot_dreamwaq import LeggedRobotDreamwaq
from legged_gym.utils.math_utils import wrap_to_pi, quat_apply
import torch.nn.functional as F
import torchvision.transforms as T

def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int], str) -> Tensor
    return torch.empty(*shape, device=device).uniform_(lower, upper)
    #return (upper - lower) * torch.rand() + lower

def _make_gaussian_kernel(sigma: float, kernel_size: int, device) -> torch.Tensor:
    """Build a normalized 2-D Gaussian kernel for conv2d."""
    if kernel_size % 2 == 0:
        kernel_size += 1  # must be odd for symmetric padding
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]  # (k, k)
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)  # (1, 1, k, k)

def _fill_rectangles(canvas, batch_idx, tops, bottoms, lefts, rights):
    """Fill axis-aligned rectangles into canvas without Python loops."""
    # Use a summed-area / prefix trick: mark corners, then cumsum twice.
    # For each rect in batch b: canvas[b, top:bottom+1, left:right+1] = 1
    # 
    # Corner-increment trick (O(n_) scatter, then 2x cumsum):
    #   +1 at (b, top,      left)
    #   -1 at (b, bottom+1, left)
    #   -1 at (b, top,      right+1)
    #   +1 at (b, bottom+1, right+1)
    N, H, W = canvas.shape
    device = canvas.device
    n_ = batch_idx.shape[0]

    r1 = bottoms + 1
    c1 = rights + 1

    def scatter(b, r, c, val):
        # Clamp out-of-bound corner updates (they fall off the edge, safe to drop)
        valid = (r < H) & (c < W)
        idx = b[valid] * H * W + r[valid] * W + c[valid]
        ones = torch.ones(n_, device=device)
        canvas.view(-1).scatter_add_(0, idx, ones[valid] * val)
        #canvas.view(-1).scatter_add_(0, idx, torch.full((valid.sum(),), val, device=device))

    ones  = torch.ones(n_, device=device)
    scatter(batch_idx, tops,  lefts,  1.)
    scatter(batch_idx, r1,    lefts, -1.)
    scatter(batch_idx, tops,  c1,    -1.)
    scatter(batch_idx, r1,    c1,     1.)

    # Two cumulative sums reconstruct the filled rectangles
    canvas.cumsum_(dim=1).cumsum_(dim=2)

class Go2DepthWaq(LeggedRobotDreamwaq):
    
    def compute_observations(self):
        self._get_forward_depth_obs()
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions ,                                                    # num_actions
        ), dim=-1)

        domain_randomization_info = torch.cat((
                    self.simulator.dr_friction_values,            # 1
                    self.simulator.dr_added_base_mass,            # 1
                    self.simulator.dr_base_com_bias,              # 3
                    self.simulator.dr_rand_push_vels[:, :2],      # 2
                    self.simulator.dr_kp_scale,                   # num_actions
                    self.simulator.dr_kd_scale                    # num_actions
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,                   # 3
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 34
        ), dim=-1)
        ## add link contact states
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # 17
                ),
                dim=-1,
            )
        ## add measured terrain heights
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
        
        # explicit info labels
        self.explicit_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel * 0.5,  # 3
            self.simulator.link_contact_states, # contact states of hips, thighs, calves, feet and base (4+4+4+4+1)=17
            torch.clip(self.simulator.feet_pos[:, :, 2] -
                torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_height_offset, -1, 1.),  # 4
        ), dim=-1)
    

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
            not self._reset_unrecoverable_gaps
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

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_camera_envs = self.cfg.env.num_camera_envs
        if hasattr(self.cfg.sensor.depth_camera_config, "resized_resolution"): 
            self.output_resolution = self.cfg.sensor.depth_camera_config.resized_resolution
            self._has_resized_resolution = True
        else:
            H, W = self.cfg.sensor.depth_camera_config.resolution 
            t, b = self.cfg.sensor.depth_camera_config.crop_top_bottom
            l, r = self.cfg.sensor.depth_camera_config.crop_left_right
            H = H - t - b
            W = W - l - r
            self.output_resolution = (H, W)
            self._has_resized_resolution = False
        self.depth_image_size = self.output_resolution[0] * self.output_resolution[1]
        self._reset_unrecoverable_gaps = getattr(self.cfg.termination, "reset_unrecoverable_gaps", False) if hasattr(self.cfg, "termination") else False
        depth_camera_config = self.cfg.sensor.depth_camera_config
        self._depth_contour_thresh  = getattr(depth_camera_config, "countour_threshold", 0.)
        self._depth_artifacts_prob  = getattr(depth_camera_config, "artifacts_prob", 0.)
        self._depth_stereo_min      = getattr(depth_camera_config, "stereo_min_distance", 0.)
        self._depth_sky_prob        = getattr(depth_camera_config, "sky_artifacts_prob", 0.)


    def _init_buffers(self):
        super()._init_buffers()
        self.gap_fall_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.depth_sensor_output = torch.zeros(
            (self.num_camera_envs, 1, *self.output_resolution),
            dtype=torch.float32,
            device=self.device,
        )
        self.env_cam_arange = torch.arange(self.num_camera_envs, device=self.device)
        self.set_latency_buffer_for_sensor()
        self.set_obs_buffers_for_component()
        self.build_depth_image_processor_buffers()

        # identify env ids for different terrain types
        if self.cfg.terrain.curriculum:
            terrain_types = self.simulator.terrain_types
            terrain_type_bounds = torch.cumsum(torch.tensor(self.cfg.terrain.terrain_proportions), dim=0) * self.cfg.terrain.num_cols
            #self.slope_env_ids = ((terrain_types >=0) & (terrain_types < terrain_type_bounds[0])).nonzero(as_tuple=False).flatten()
            self.stairs_env_ids = ((terrain_types >= terrain_type_bounds[2]) & (terrain_types < terrain_type_bounds[3])).nonzero(as_tuple=False).flatten()
            self.discrete_env_ids = ((terrain_types >= terrain_type_bounds[3]) & (terrain_types < terrain_type_bounds[4])).nonzero(as_tuple=False).flatten()
            self.stepping_stones_env_ids = ((terrain_types >= terrain_type_bounds[4]) & (terrain_types < terrain_type_bounds[5])).nonzero(as_tuple=False).flatten()
            self.gaps_env_ids = ((terrain_types >= terrain_type_bounds[5]) & (terrain_types < terrain_type_bounds[6])).nonzero(as_tuple=False).flatten()
            self.pits_env_ids = ((terrain_types >= terrain_type_bounds[6]) & (terrain_types < terrain_type_bounds[7])).nonzero(as_tuple=False).flatten()
            self.high_platform_env_ids = ((terrain_types >= terrain_type_bounds[7]) & (terrain_types < terrain_type_bounds[8])).nonzero(as_tuple=False).flatten()
            self.high_platform_gaps_env_ids = ((terrain_types >= terrain_type_bounds[8]) & (terrain_types < terrain_type_bounds[9])).nonzero(as_tuple=False).flatten()
            # identify env ids for all heading command (others have fixed heading command specified in config)
            self.all_heading_env_ids = torch.cat((
                #self.slope_env_ids,
                self.stairs_env_ids,
                self.discrete_env_ids,
                self.gaps_env_ids,
                self.pits_env_ids, # elementary terrains with all heading commands
            ))
            # identity termination base height for high_platform_gaps terrain
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.high_platform_gaps_termination_height = eval(
                self.cfg.terrain.terrain_curriculum_difficulty["high_platform_gaps_params"]["high_platform_height"])
            
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.platform_size = self.cfg.terrain.platform_size
            self.pit_depth = eval(self.cfg.terrain.terrain_curriculum_difficulty["pit_depth"])
        else:
            if "depth" in self.cfg.terrain.terrain_kwargs:
                self.pit_depth = self.cfg.terrain.terrain_kwargs["depth"]
    
    def _resample_commands(self, env_ids) -> None:
        if not self._reset_unrecoverable_gaps:
            super()._resample_commands(env_ids)
            return
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            if self.cfg.terrain.curriculum:
                # override heading command of some envs with all range heading command
                env_ids_for_heading = torch.tensor([env_id for env_id in env_ids if env_id in self.all_heading_env_ids], device=self.device)
                if len(env_ids_for_heading) > 0:
                    self.commands[env_ids_for_heading, 3] = torch_rand_float(-3.14, 3.14, (len(env_ids_for_heading), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if np.random.rand() < self.cfg.commands.zero_cmd_prob:
            self.commands[env_ids, :3] *= 0.0  # set command to zero with some probability, to encourage the robot to learn to stand still
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self._reset_unrecoverable_gaps:
            super()._update_command_curriculum(env_ids)
            return
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            # only increase upper bound of forward velocity command
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
    def _pre_sim_step(self, actions):
        self.depth_sensor_obs_refreshed = False
        return super()._pre_sim_step(actions)

    def step(self, actions):
        return *super().step(actions), self.depth_sensor_output

    def get_observations(self):
        return *super().get_observations(), self.depth_sensor_output

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, explicit_labels, next_state, _, _, _, depth_sensor_output = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, obs_history, explicit_labels, next_state, depth_sensor_output

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.gap_fall_counter[env_ids] = 0
        difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
        self.pit_depth = eval(self.cfg.terrain.terrain_curriculum_difficulty["pit_depth"])

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._resample_sensor_latency_if_needed()

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environment ids
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
        return noise_vec

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # refer to https://ieeexplore.ieee.org/document/11112615
        heading_error = torch.abs(self.heading - self.commands[:, 3])
        heading_coef = (1 + torch.cos(heading_error)) / 2
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.simulator.base_lin_vel[:, 0])
        # double the weight of y velocity error to discourage y axis drifting
        lin_vel_y_error = 2 * torch.square(self.commands[:, 1] - self.simulator.base_lin_vel[:, 1])
        lin_vel_error = lin_vel_x_error + lin_vel_y_error
        # add heading_coef to make the reward smaller when the robot is facing away from the commanded direction
        # thus to encourage the robot to walk across the terrain in the commanded direction
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) * heading_coef
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.simulator.link_contact_forces[:, self.simulator.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
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
    
    def _reward_foot_clearance_terrain_aware(self):
        """
        Encourage swing feet to reach a terrain-aware desired height,
        while softly discouraging excessive swing height.

        Assumes:
            self.simulator.feet_pos           : (N, 4, 3)
            self.simulator.feet_vel           : (N, 4, 3)
            self.simulator._height_around_feet: (N, 4, 3, 3) or (N, 4, 9)

        Uses:
            - terrain-aware target height
            - horizontal foot velocity weighting (same style as original reward)
            - excess-height penalty to prevent over-swinging
        """

        feet_z = self.simulator.feet_pos[:, :, 2]                       # (N,4)
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)  # (N,4)

        # Flatten 3x3 terrain patch if needed, then take local max height near each foot
        h_patch = self.simulator._height_around_feet
        if h_patch.ndim == 4:   # (N,4,3,3)
            h_patch = h_patch.view(h_patch.shape[0], h_patch.shape[1], -1)  # (N,4,9)

        local_terrain_h = torch.max(h_patch, dim=-1)[0]                # (N,4)

        # Terrain-aware desired foot height
        z_des = (
            self.cfg.rewards.foot_clearance_target
            + self.cfg.rewards.foot_height_offset
            + local_terrain_h
        )                                                               # (N,4)

        # Main tracking error: encourage feet to reach desired terrain-aware height
        track_err = torch.square(feet_z - z_des)                        # (N,4)

        # Soft over-swing penalty: only penalize when foot goes too far above desired height
        # Margin gives some freedom to overshoot a little during learning
        excess_margin = 0.04  # [m], tune: 0.03 - 0.06
        excess = torch.relu(feet_z - (z_des + excess_margin))           # (N,4)
        # excess = F.softplus(feet_z - (z_des + excess_margin))           # (N,4)
        excess_err = torch.square(excess)

        # Weight excess penalty less than main tracking term
        excess_weight = 0.25  # tune: 0.1 - 0.5

        total_err = torch.sum(
            foot_vel_xy_norm * (track_err + excess_weight * excess_err),
            dim=-1
        )                                                               # (N,)

        return torch.exp(-total_err / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _reward_feet_near_edge(self):
        """ Penalize feet being too close to the edge of a terrain
        """
        feet_near_edge = self.simulator.calc_feet_near_edge()
        feet_contact = self.feet_max_force_z > 10.0
        # print(f"feet near edge: {torch.sum(feet_near_edge)}")
        return torch.sum(feet_near_edge * feet_contact, dim=-1)

    def _reward_base_up_pit(self):
        base_height = torch.clamp(self.simulator.base_pos[:, 2] - self.simulator.env_origins[:, 2], min=0.0)
        error = torch.square(base_height - self.cfg.rewards.base_height_target - self.pit_depth)
        return torch.exp(-error / self.cfg.rewards.base_up_pit_sigma)

    def get_failure_idx(self):
        return self.reset_buf * ~self.time_out_buf

    #### DEPTH SECTION ####
    def build_depth_image_processor_buffers(self):
        if hasattr(self.cfg.sensor.depth_camera_config, "resized_resolution"):
            self.depth_resize_transform = T.Resize(
                   self.cfg.sensor.depth_camera_config.resized_resolution,
                    interpolation= T.InterpolationMode.BICUBIC,
                )
        self.contour_detection_kernel = torch.zeros(
                (8, 1, 3, 3),
                dtype= torch.float32,
                device= self.device,
            )
            # emperical values to be more sensitive to vertical edges
        self.contour_detection_kernel[0, :, 1, 1] = 0.5
        self.contour_detection_kernel[0, :, 0, 0] = -0.5
        self.contour_detection_kernel[1, :, 1, 1] = 0.1
        self.contour_detection_kernel[1, :, 0, 1] = -0.1
        self.contour_detection_kernel[2, :, 1, 1] = 0.5
        self.contour_detection_kernel[2, :, 0, 2] = -0.5
        self.contour_detection_kernel[3, :, 1, 1] = 1.2
        self.contour_detection_kernel[3, :, 1, 0] = -1.2
        self.contour_detection_kernel[4, :, 1, 1] = 1.2
        self.contour_detection_kernel[4, :, 1, 2] = -1.2
        self.contour_detection_kernel[5, :, 1, 1] = 0.5
        self.contour_detection_kernel[5, :, 2, 0] = -0.5
        self.contour_detection_kernel[6, :, 1, 1] = 0.1
        self.contour_detection_kernel[6, :, 2, 1] = -0.1
        self.contour_detection_kernel[7, :, 1, 1] = 0.5
        self.contour_detection_kernel[7, :, 2, 2] = -0.5

    def set_obs_buffers_for_component(self):
        buffer_length = int(self.cfg.sensor.depth_camera_config.latency_range[1] / self.dt) + 1
        # use super().get_obs_segment_from_components() to get the obs shape to prevent post processing
        # overrides the buffer shape
        H, W = self.output_resolution
        obs_buffer = torch.zeros(
            (
                buffer_length,
                self.num_camera_envs,
                H, W
            ),
            dtype= torch.float32,
            device= self.device,
        )
        self.depth_sensor_obs_buffer = obs_buffer
        self.depth_sensor_obs_refreshed = False
    
    def set_latency_buffer_for_sensor(self):
        latency_buffer = torch_rand_float(
            self.cfg.sensor.depth_camera_config.latency_range[0],
            self.cfg.sensor.depth_camera_config.latency_range[1],
            (self.num_camera_envs,),
            device= self.device,
        )
        # using setattr to set the buffer
        self.depth_sensor_latency_buffer = latency_buffer
        self.depth_sensor_delayed_frames = torch.zeros_like(latency_buffer, dtype= torch.long, device= self.device)

    def _resample_sensor_latency_if_needed(self):
        resampling_time = getattr(self.cfg.sensor.depth_camera_config, "latency_resampling_time", self.dt)
        resample_mask = ((self.episode_length_buf % int(resampling_time / self.dt)) == 0)[:self.num_camera_envs]

        num_resample = resample_mask.sum()
        if num_resample == 0:
            return
        new_latencies = torch_rand_float(
            self.cfg.sensor.depth_camera_config.latency_range[0],
            self.cfg.sensor.depth_camera_config.latency_range[1],
            (num_resample.item(),),
            device=self.device,
        ).squeeze(-1)  # depending on torch_rand_float's output shape

        self.depth_sensor_latency_buffer[:self.num_camera_envs][resample_mask] = new_latencies

        #if not resample_mask.any(): 
        #    return
        #new_latencies = torch_rand_float(
        #    self.cfg.sensor.depth_camera_config.latency_range[0],
        #    self.cfg.sensor.depth_camera_config.latency_range[1],
        #    (self.num_camera_envs,),
        #    device=self.device,
        #)
        #self.depth_sensor_latency_buffer[resample_mask] = new_latencies[resample_mask]

        #resample_env_ids = (self.episode_length_buf %  == 0).nonzero(as_tuple= False).flatten()
        #if len(resample_env_ids) > 0:
        #    self.depth_sensor_latency_buffer[resample_env_ids] = torch_rand_float(
        #        self.cfg.sensor.depth_camera_config.latency_range[0],
        #        self.cfg.sensor.depth_camera_config.latency_range[1],
        #        (len(resample_env_ids),),
        #        device= self.device,
        #    )
    
    def _reset_buffers(self, env_ids):
        return_ = super()._reset_buffers(env_ids)
        self.depth_sensor_obs_buffer[:, env_ids] = 0
        self.depth_sensor_obs_refreshed = False
        self.depth_sensor_delayed_frames[env_ids] = 0

    ######################## IMAGE PROSS ############################################

    def _add_depth_contour(self, depth_images):
        mask =  F.max_pool2d(
            torch.abs(F.conv2d(depth_images, self.contour_detection_kernel, padding= 1)).max(dim= -3, keepdim= True)[0],
            kernel_size= self.cfg.noise.forward_depth.contour_detection_kernel_size,
            stride= 1,
            padding= self.cfg.noise.forward_depth.contour_detection_kernel_size // 2,
        ) > self.cfg.noise.forward_depth.contour_threshold
        depth_images[mask] = 0.
        return depth_images
    
    # def _add_depth_artifacts(self, depth_images,
    #         artifacts_prob,
    #         artifacts_height_mean_std,
    #         artifacts_width_mean_std,
    #     ):
    #     """ Simulate artifacts from stereo depth camera. In the final artifacts_mask, where there
    #     should be an artifacts, the mask is 1.
    #     """
    #     N, H, W = depth_images.shape
    #     def _clip(x, dim):
    #         return torch.clip(x, 0., (H, W)[dim])

    #     # random patched artifacts
    #     artifacts_mask = torch_rand_float_3d(
    #         0., 1.,
    #         (N, H, W),
    #         device= self.device,
    #     ) < artifacts_prob
    #     artifacts_mask = artifacts_mask & (depth_images[:] > 0.)
    #     artifacts_coord = torch.nonzero(artifacts_mask).to(torch.float32) # (n_, 3) n_ <= N * H * W
    #     artifcats_size = (
    #         torch.clip(
    #             artifacts_height_mean_std[0] + torch.randn(
    #                 (artifacts_coord.shape[0],),
    #                 device= self.device,
    #             ) * artifacts_height_mean_std[1],
    #             0., H,
    #         ),
    #         torch.clip(
    #             artifacts_width_mean_std[0] + torch.randn(
    #                 (artifacts_coord.shape[0],),
    #                 device= self.device,
    #             ) * artifacts_width_mean_std[1],
    #             0., W,
    #         ),
    #     ) # (n_,), (n_,)
    #     artifacts_top_left = (
    #         _clip(artifacts_coord[:, 1] - artifcats_size[0] / 2, 0),
    #         _clip(artifacts_coord[:, 2] - artifcats_size[1] / 2, 1),
    #     )
    #     artifacts_bottom_right = (
    #         _clip(artifacts_coord[:, 1] + artifcats_size[0] / 2, 0),
    #         _clip(artifacts_coord[:, 2] + artifcats_size[1] / 2, 1),
    #     )
    #     @torch.no_grad()
    #     def form_artifacts(
    #             H, W, # image resolution
    #             tops, bottoms, # artifacts positions (in pixel) shape (n_,)
    #             lefts, rights,
    #         ):
    #         """ Paste an artifact to the depth image.
    #         NOTE: Using the paradigm of spatial transformer network to build the artifacts of the
    #         entire depth image.
    #         """
    #         batch_size = tops.shape[0]
    #         tops, bottoms = tops[:, None, None], bottoms[:, None, None]
    #         lefts, rights = lefts[:, None, None], rights[:, None, None]

    #         # build the source patch
    #         source_patch = torch.zeros((batch_size, 1, 25, 25), device= self.device)
    #         source_patch[:, :, 1:24, 1:24] = 1.

    #         # build the grid
    #         grid = torch.zeros((batch_size, H, W, 2), device= self.device)
    #         grid[..., 0] = torch.linspace(-1, 1, W, device= self.device).view(1, 1, W)
    #         grid[..., 1] = torch.linspace(-1, 1, H, device= self.device).view(1, H, 1)
    #         grid[..., 0] = (grid[..., 0] * W + W - rights - lefts) / (rights - lefts)
    #         grid[..., 1] = (grid[..., 1] * H + H - bottoms - tops) / (bottoms - tops)

    #         # sample using the grid and form the artifacts for the entire depth image
    #         artifacts = torch.clip(
    #             F.grid_sample(
    #                 source_patch,
    #                 grid,
    #                 mode= "bilinear",
    #                 padding_mode= "zeros",
    #                 align_corners= False,
    #             ).sum(dim= 0).view(H, W),
    #             0, 1,
    #         )

    #         return artifacts

    #     for i in range(N):
    #         artifacts_mask = form_artifacts(
    #             H, W,
    #             artifacts_top_left[0][artifacts_coord[:, 0] == i],
    #             artifacts_bottom_right[0][artifacts_coord[:, 0] == i],
    #             artifacts_top_left[1][artifacts_coord[:, 0] == i],
    #             artifacts_bottom_right[1][artifacts_coord[:, 0] == i],
    #         )
    #         depth_images[i] *= (1 - artifacts_mask)

    #     return depth_images

    
    

    # def _add_depth_artifacts(
    #     self,
    #     depth_images,
    #     artifacts_prob,
    #     artifacts_height_mean_std,
    #     artifacts_width_mean_std,
    # ):
    #     """Simulate artifacts from stereo depth camera using direct rectangle rendering."""

    #     def _fill_rectangles(canvas, batch_idx, tops, bottoms, lefts, rights):
    #         """Fill axis-aligned rectangles into canvas without Python loops."""
    #         # Use a summed-area / prefix trick: mark corners, then cumsum twice.
    #         # For each rect in batch b: canvas[b, top:bottom+1, left:right+1] = 1
    #         # 
    #         # Corner-increment trick (O(n_) scatter, then 2x cumsum):
    #         #   +1 at (b, top,      left)
    #         #   -1 at (b, bottom+1, left)
    #         #   -1 at (b, top,      right+1)
    #         #   +1 at (b, bottom+1, right+1)
    #         N, H, W = canvas.shape
    #         device = canvas.device
    #         n_ = batch_idx.shape[0]

    #         r1 = bottoms + 1
    #         c1 = rights + 1

    #         def scatter(b, r, c, val):
    #             # Clamp out-of-bound corner updates (they fall off the edge, safe to drop)
    #             valid = (r < H) & (c < W)
    #             idx = b[valid] * H * W + r[valid] * W + c[valid]
    #             canvas.view(-1).scatter_add_(0, idx, torch.full((valid.sum(),), val, device=device))

    #         ones  = torch.ones(n_, device=device)
    #         scatter(batch_idx, tops,  lefts,  1.)
    #         scatter(batch_idx, r1,    lefts, -1.)
    #         scatter(batch_idx, tops,  c1,    -1.)
    #         scatter(batch_idx, r1,    c1,     1.)

    #         # Two cumulative sums reconstruct the filled rectangles
    #         canvas.cumsum_(dim=1).cumsum_(dim=2)
        
    #     N, H, W = depth_images.shape
    #     device = self.device

    #     # --- Sample seed pixels via Bernoulli (avoids full float random + compare) ---
    #     artifacts_mask = torch.rand(N, H, W, device=device) < artifacts_prob
    #     artifacts_mask &= depth_images > 0.

    #     artifacts_coord = artifacts_mask.nonzero()
    #     n_ = artifacts_coord.shape[0]

    #     if n_ == 0:
    #         return depth_images

    #     # --- Sample rectangle sizes ---
    #     h_half = (
    #         artifacts_height_mean_std[0] + torch.randn(n_, device=device) * artifacts_height_mean_std[1]
    #     ).clamp(0., H) / 2  # (n_,)

    #     w_half = (
    #         artifacts_width_mean_std[0] + torch.randn(n_, device=device) * artifacts_width_mean_std[1]
    #     ).clamp(0., W) / 2  # (n_,)

    #     # Seed coords (integer)
    #     batch_idx = artifacts_coord[:, 0]  # (n_,)
    #     cy        = artifacts_coord[:, 1].float()
    #     cx        = artifacts_coord[:, 2].float()

    #     # Bounding boxes, clamped to image bounds
    #     tops    = (cy - h_half).clamp(0., H - 1).long()   # (n_,)
    #     bottoms = (cy + h_half).clamp(0., H - 1).long()
    #     lefts   = (cx - w_half).clamp(0., W - 1).long()
    #     rights  = (cx + w_half).clamp(0., W - 1).long()

    #     # --- Render all rectangles directly into a mask ---
    #     # Build combined artifact mask (N, H, W) by filling rectangles
    #     combined = torch.zeros(N, H, W, device=device)
    #     _fill_rectangles(combined, batch_idx, tops, bottoms, lefts, rights)

    #     depth_images *= (1 - combined.clamp(0., 1.))
    #     return depth_images

    def _add_depth_artifacts(
        self,
        depth_images,
        artifacts_prob,
        artifacts_height_mean_std,
        artifacts_width_mean_std,
        blur_sigma: float = 1.5,       # Gaussian falloff
        blur_kernel_size: int = 11, 
    ):
        
        """Simulate artifacts from stereo depth camera using direct rectangle rendering."""
        N, H, W = depth_images.shape
        device = self.device

        # --- Sample seed pixels via Bernoulli (avoids full float random + compare) ---
        #artifacts_mask = torch.bernoulli(
        #    torch.empty(N, H, W, device=device).fill_(artifacts_prob)
        #).bool()
        #artifacts_mask &= depth_images > 0.
        artifacts_mask = torch.rand(N, H, W, device=device) < artifacts_prob

        artifacts_coord = artifacts_mask.nonzero()  # (n_, 3), int64 — skip .float() for now
        n_ = artifacts_coord.shape[0]

        if n_ == 0:
            return depth_images

        # --- Sample rectangle sizes ---
        h_half = (
            artifacts_height_mean_std[0]
            + torch.randn(n_, device=device) * artifacts_height_mean_std[1]
        ).clamp(0., H).mul_(0.5)  # (n_,)

        w_half = (
            artifacts_width_mean_std[0]
            + torch.randn(n_, device=device) * artifacts_width_mean_std[1]
        ).clamp(0., W).mul_(0.5)  # (n_,)

        # Seed coords (integer)
        batch_idx = artifacts_coord[:, 0]  # (n_,)
        cy        = artifacts_coord[:, 1].float()
        cx        = artifacts_coord[:, 2].float()

        # Bounding boxes, clamped to image bounds
        tops    = (cy - h_half).clamp(0., H - 1).long()   # (n_,)
        bottoms = (cy + h_half).clamp(0., H - 1).long()
        lefts   = (cx - w_half).clamp(0., W - 1).long()
        rights  = (cx + w_half).clamp(0., W - 1).long()

        # --- Render all rectangles directly into a mask ---
        # Build combined artifact mask (N, H, W) by filling rectangles
        combined = torch.zeros(N, H, W, device=device)
        _fill_rectangles(combined, batch_idx, tops, bottoms, lefts, rights)
        combined.clamp_(0., 1.)

        #Cache Gaussian kernel — recompute only if params changed
        cache_key = (blur_sigma, blur_kernel_size)
        if getattr(self, '_artifact_blur_cache_key', None) != cache_key:
            self._artifact_blur_kernel   = _make_gaussian_kernel(blur_sigma, blur_kernel_size, device)
            self._artifact_blur_pad      = blur_kernel_size // 2
            self._artifact_blur_cache_key = cache_key

        #kernel = _make_gaussian_kernel(blur_sigma, blur_kernel_size, device)
        #pad = blur_kernel_size // 2
        blurred = F.conv2d(
            F.pad(combined.unsqueeze(1), (self._artifact_blur_pad,) * 4, mode="reflect"),          # (N, 1, H, W)
            self._artifact_blur_kernel,                         # (1, 1, k, k)
            padding=0,
        ).squeeze(1)                        # (N, H, W)
        depth_images *= (1. - blurred)

        return depth_images

    
    def _recognize_top_down_too_close(self, too_close_mask):
        """ Based on real D435i image pattern, there are two situations when pixels are too close
        Whether there is too-close pixels all the way across the image vertically.
        """
        # vertical_all_too_close = too_close_mask.all(dim= 2, keepdim= True)
        vertical_too_close = too_close_mask.sum(dim= -2, keepdim= True) > (too_close_mask.shape[-2] * 0.6)
        return vertical_too_close
    
    def _add_depth_stereo(self, depth_images):
        """ Simulate the noise from the depth limit of the stereo camera. """
        N, H, W = depth_images.shape
        far_mask = depth_images > self.cfg.sensor.depth_camera_config.stereo_far_distance
        too_close_mask = depth_images < self.cfg.sensor.depth_camera_config.stereo_min_distance
        near_mask = (~far_mask) & (~too_close_mask)
        
        n_far = far_mask.sum()
        if far_mask.sum() > 0:
            depth_images[far_mask] += torch.empty(n_far, device=self.device).uniform_(
                0., self.cfg.sensor.depth_camera_config.stereo_far_noise_std
            )
        # add noise to the far points
        #far_noise = torch_rand_float(
        #    0., self.cfg.sensor.depth_camera_config.stereo_far_noise_std,
        #    (N, H, W),
        #    device= self.device,
        #)
        #far_noise = far_noise * far_mask
        #depth_images += far_noise

        # add noise to the near points
        #near_noise = torch_rand_float(
        #    0., self.cfg.sensor.depth_camera_config.stereo_near_noise_std,
        #    (N, H, W),
        #    device= self.device,
        #)
        #near_noise = near_noise * near_mask
        #depth_images += near_noise

        n_near = near_mask.sum()
        if n_near > 0:
            depth_images[near_mask] += torch.empty(n_near, device=self.device).uniform_(
                0., self.cfg.sensor.depth_camera_config.stereo_near_noise_std
            )

        # add artifacts to the too close points
        vertical_block_mask = self._recognize_top_down_too_close(too_close_mask)
        full_block_mask = vertical_block_mask & too_close_mask
        half_block_mask = (~vertical_block_mask) & too_close_mask
        # add artifacts where vertical pixels are all too close
        #for pixel_value in random.sample(
        #        self.cfg.sensor.depth_camera_config.stereo_full_block_values,
        #        len(self.cfg.sensor.depth_camera_config.stereo_full_block_values),
        #    ):
        for pixel_value in self.cfg.sensor.depth_camera_config.stereo_full_block_values:
            artifacts_buffer = self._add_depth_artifacts(torch.ones_like(depth_images),
                self.cfg.sensor.depth_camera_config.stereo_full_block_artifacts_prob,
                self.cfg.sensor.depth_camera_config.stereo_full_block_height_mean_std,
                self.cfg.sensor.depth_camera_config.stereo_full_block_width_mean_std,
            )
            depth_images[full_block_mask] = ((1 - artifacts_buffer) * pixel_value)[full_block_mask]
        # add artifacts where not all the same vertical pixels are too close
        n_half = half_block_mask.sum()
        if n_half > 0:
            spark = torch.bernoulli(
                torch.full((n_half,), self.cfg.sensor.depth_camera_config.stereo_half_block_spark_prob, device=self.device)
            )
            depth_images[half_block_mask] = spark * self.cfg.sensor.depth_camera_config.stereo_half_block_value
            #half_block_spark = torch.rand(N, H, W, device=self.device) < self.cfg.sensor.depth_camera_config.stereo_half_block_spark_prob
            #depth_images[half_block_mask] = (half_block_spark.to(torch.float32) * self.cfg.sensor.depth_camera_config.stereo_half_block_value)[half_block_mask]

        return depth_images
    
    def _recognize_top_down_seeing_sky(self, too_far_mask):
        N, H, W = too_far_mask.shape
        # whether there is too-far pixels with all pixels above it too-far
        num_too_far_above = too_far_mask.cumsum(dim= -2)
        all_too_far_above_threshold = torch.arange(H, device= self.device).view(1, H, 1)
        return num_too_far_above > all_too_far_above_threshold # (N, 1, H, W) mask
    
    def _add_sky_artifacts(self, depth_images):
        """ Incase something like ceiling pattern or stereo failure happens. """
        N, H, W         = depth_images.shape
        
        to_sky_mask     = self._recognize_top_down_seeing_sky(depth_images > self.cfg.sensor.depth_camera_config.sky_artifacts_far_distance)
        isinf_mask      = depth_images.isinf()
        sky_finite_mask = to_sky_mask & ~isinf_mask
        sky_inf_mask    = to_sky_mask & isinf_mask
        
        # add artifacts to the regions where they are seemingly pointing to sky
        for pixel_value in self.cfg.sensor.depth_camera_config.sky_artifacts_values:
            artifacts_buffer = self._add_depth_artifacts(torch.ones_like(depth_images),
                self.cfg.sensor.depth_camera_config.sky_artifacts_prob,
                self.cfg.sensor.depth_camera_config.sky_artifacts_height_mean_std,
                self.cfg.sensor.depth_camera_config.sky_artifacts_width_mean_std,
            )
            depth_images[sky_finite_mask] *= artifacts_buffer[sky_finite_mask]
            depth_images[sky_inf_mask & (artifacts_buffer < 1)] = 0.
            depth_images[to_sky_mask] += ((1 - artifacts_buffer) * pixel_value)[to_sky_mask]
        
        return depth_images

    def _crop_depth_images(self, depth_images):
        H, W = depth_images.shape[-2:]
        t, b = self.cfg.sensor.depth_camera_config.crop_top_bottom
        l, r = self.cfg.sensor.depth_camera_config.crop_left_right 
        return depth_images[...,
            t: H - b,
            l: W - r,
        ]

    def _normalize_depth_images(self, depth_images):
        # normalize depth image to (0.0, 1.0)
        near, far = self.cfg.sensor.depth_camera_config.near_clip, self.cfg.sensor.depth_camera_config.far_clip
        return ( depth_images.clamp(near, far) - near ) / ( far - near )
    
    @torch.no_grad()
    def _process_depth_image(self, depth_images):
        # depth_images length N list with shape (H, W)
        # reverse the negative depth (according to the document)
        # depth_images_ = torch.stack(depth_images).unsqueeze(1).contiguous().detach().clone() * -1
        depth_images_ = depth_images.clone().squeeze(1)
        if self._depth_artifacts_prob > 0.:
            depth_images_ = self._add_depth_contour(depth_images_)
        if self._depth_artifacts_prob > 0.:
            depth_images_ = self._add_depth_artifacts(depth_images_,
                self.cfg.sensor.depth_camera_config.artifacts_prob,
                self.cfg.sensor.depth_camera_config.artifacts_height_mean_std,
                self.cfg.sensor.depth_camera_config.artifacts_width_mean_std,
            )
        if self._depth_stereo_min > 0.:
            depth_images_ = self._add_depth_stereo(depth_images_)
        if self._depth_sky_prob > 0.:
            depth_images_ = self._add_sky_artifacts(depth_images_)
        depth_images_ = self._normalize_depth_images(depth_images_)
        depth_images_ = self._crop_depth_images(depth_images_)
        if self._has_resized_resolution:
            depth_images_ = self.depth_resize_transform(depth_images_)
        depth_images_ = depth_images_.clamp(0, 1)
        return depth_images_

    def _get_forward_depth_obs(self):
        if not self.depth_sensor_obs_refreshed:
            self.depth_sensor_obs_buffer = torch.roll(self.depth_sensor_obs_buffer, shifts=-1, dims=0)
            self.depth_sensor_obs_buffer[-1] = self._process_depth_image(
                self.simulator.depth_images
            )
            #self.depth_sensor_obs_buffer[:-1] = self.depth_sensor_obs_buffer[1:].clone()
            #self.depth_sensor_obs_buffer[-1] = self._process_depth_image(self.simulator.depth_images)

            #to be optimized
            #self.depth_sensor_obs_buffer = torch.cat([
            #    self.depth_sensor_obs_buffer[1:],
            #    self._process_depth_image(self.simulator.depth_images).unsqueeze(0),
            #], dim= 0)

            delay_refresh_mask = (self.episode_length_buf[:self.num_camera_envs] % int(self.cfg.sensor.depth_camera_config.refresh_duration / self.dt)) == 0
            # NOTE: if the delayed frames is greater than the last frame, the last image should be used.
            frame_select = (self.depth_sensor_latency_buffer / self.dt).long()
            next_frame = self.depth_sensor_delayed_frames + 1
            
            self.depth_sensor_delayed_frames = torch.where(
                delay_refresh_mask,
                torch.minimum(
                    frame_select,
                    next_frame,
                ),
                next_frame,
            ).clamp_(0, self.depth_sensor_obs_buffer.shape[0])

            self.depth_sensor_output.copy_(
                self.depth_sensor_obs_buffer[
                    -self.depth_sensor_delayed_frames,
                    self.env_cam_arange,
                ].unsqueeze(1)
            )
            self.depth_sensor_obs_refreshed = True
        #return self.depth_sensor_output
    #if not hasattr(self.cfg.sensor, "forward_camera") or privileged:
    #    return super()._get_forward_depth_obs(privileged).reshape(self.num_envs, -1)


#updated