import torch
import torch.nn.functional as F


class DepthMixin:
    def _init_depth_processing(self):
        if not self.cfg.sensor.add_depth:
            self.depth_sensor_output = None
            return

        cfg = self.cfg.sensor.depth_camera_config
        if hasattr(cfg, "resized_resolution"):
            self.depth_output_resolution = tuple(cfg.resized_resolution)
        else:
            height, width = cfg.resolution
            top, bottom = cfg.crop_top_bottom
            left, right = cfg.crop_left_right
            self.depth_output_resolution = (
                height - top - bottom,
                width - left - right,
            )

        self.depth_sensor_output = torch.zeros(
            (
                self.num_envs,
                cfg.num_history,
                *self.depth_output_resolution,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        buffer_length = max(2, int(cfg.latency_range[1] / self.dt) + 2)
        self.depth_sensor_obs_buffer = torch.zeros(
            (
                buffer_length,
                self.num_envs,
                *self.depth_output_resolution,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.depth_sensor_obs_write_idx = 0
        self.depth_sensor_latency = torch.empty(
            self.num_envs, device=self.device
        ).uniform_(*cfg.latency_range)
        self.depth_sensor_delayed_frames = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.depth_sensor_obs_refreshed = False
        self.depth_env_ids = torch.arange(
            self.num_envs, device=self.device
        )
        self._sky_artifact_values = torch.as_tensor(
            cfg.sky_artifacts_values,
            device=self.device,
            dtype=torch.float32,
        )

    def get_depth_observations(self):
        return self.depth_sensor_output

    def _pre_depth_step(self):
        if self.cfg.sensor.add_depth:
            self.depth_sensor_obs_refreshed = False

    def _update_depth_observations(self):
        if not self.cfg.sensor.add_depth or self.depth_sensor_obs_refreshed:
            return

        cfg = self.cfg.sensor.depth_camera_config
        processed = self._process_depth_images(
            self.simulator.depth_images[:, 0]
        )

        # Keep the latency buffer in-place and advance a write index instead
        # of rolling the whole tensor every depth update.
        write_idx = self.depth_sensor_obs_write_idx
        self.depth_sensor_obs_buffer[write_idx] = processed
        buffer_length = self.depth_sensor_obs_buffer.shape[0]
        self.depth_sensor_obs_write_idx = (write_idx + 1) % buffer_length

        refresh_steps = max(1, int(cfg.refresh_duration / self.dt))
        refresh = (
            self.episode_length_buf % refresh_steps
        ) == 0
        requested_delay = (self.depth_sensor_latency / self.dt).long()
        self.depth_sensor_delayed_frames = torch.where(
            refresh,
            requested_delay,
            self.depth_sensor_delayed_frames + 1,
        ).clamp(0, buffer_length - 1)

        # Delayed-frame lookup is relative to the most recently written slot.
        # The output tensor below still keeps the downstream-facing order
        # [latest, previous, older, ...].
        latest_idx = (self.depth_sensor_obs_write_idx - 1) % buffer_length
        frame_indices = (
            latest_idx - self.depth_sensor_delayed_frames
        ) % buffer_length
        latest = self.depth_sensor_obs_buffer[
            frame_indices, self.depth_env_ids
        ]
        if self.depth_sensor_output.shape[1] > 1:
            self.depth_sensor_output[:, 1:] = self.depth_sensor_output[
                :, :-1
            ].clone()
        self.depth_sensor_output[:, 0] = latest
        self.depth_sensor_obs_refreshed = True

    def _resample_depth_latency(self):
        if not self.cfg.sensor.add_depth:
            return
        cfg = self.cfg.sensor.depth_camera_config
        resample_steps = max(1, int(cfg.latency_resampling_time / self.dt))
        mask = (
            self.episode_length_buf % resample_steps
        ) == 0
        if mask.any():
            new_latency = torch.empty_like(self.depth_sensor_latency).uniform_(
                *cfg.latency_range
            )
            self.depth_sensor_latency[mask] = new_latency[mask]

    def _reset_depth_buffers(self, env_ids):
        if not self.cfg.sensor.add_depth:
            return
        if env_ids.numel() == 0:
            return
        self.depth_sensor_obs_buffer[:, env_ids] = 0.0
        self.depth_sensor_output[env_ids] = 0.0
        self.depth_sensor_delayed_frames[env_ids] = 0
        cfg = self.cfg.sensor.depth_camera_config
        self.depth_sensor_latency[env_ids] = torch.empty(
            env_ids.numel(), device=self.device
        ).uniform_(*cfg.latency_range)

    @torch.no_grad()
    def _process_depth_images(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        depth = depth.clone()
        depth = self._add_stereo_noise(depth)
        depth = self._add_sky_artifacts(depth)
        depth = depth.clamp(cfg.near_clip, cfg.far_clip)
        depth = (depth - cfg.near_clip) / max(
            cfg.far_clip - cfg.near_clip, 1e-6
        )

        top, bottom = cfg.crop_top_bottom
        left, right = cfg.crop_left_right
        height, width = depth.shape[-2:]
        depth = depth[
            ...,
            top : height - bottom if bottom else height,
            left : width - right if right else width,
        ]
        if tuple(depth.shape[-2:]) != self.depth_output_resolution:
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=self.depth_output_resolution,
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
        return depth.clamp(0.0, 1.0)

    def _add_stereo_noise(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        if cfg.stereo_min_distance <= 0:
            return depth

        far_mask = depth > cfg.stereo_far_distance
        near_mask = (depth >= cfg.stereo_min_distance) & ~far_mask
        if far_mask.any() and cfg.stereo_far_noise_std > 0:
            far_noise = torch.randn_like(depth).abs() * cfg.stereo_far_noise_std
            depth = torch.where(far_mask, depth + far_noise, depth)
        if near_mask.any() and cfg.stereo_near_noise_std > 0:
            near_noise = torch.randn_like(depth) * cfg.stereo_near_noise_std
            depth = torch.where(near_mask, depth + near_noise, depth)

        too_close = depth < cfg.stereo_min_distance
        if too_close.any():
            spark = torch.rand_like(depth) < cfg.stereo_half_block_spark_prob
            depth[too_close & spark] = cfg.far_clip
            depth[too_close & ~spark] = cfg.near_clip
        return depth

    def _add_sky_artifacts(self, depth):
        cfg = self.cfg.sensor.depth_camera_config
        if cfg.sky_artifacts_prob <= 0:
            return depth
        sky = depth > cfg.sky_artifacts_far_distance
        artifacts = (
            torch.rand_like(depth) < cfg.sky_artifacts_prob
        ) & sky
        if artifacts.any():
            values = self._sky_artifact_values.to(dtype=depth.dtype)
            choices = torch.randint(
                values.numel(), depth.shape, device=depth.device
            )
            depth[artifacts] = values[choices[artifacts]]
        return depth