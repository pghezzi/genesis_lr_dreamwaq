from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

from rsl_rl.algorithms import PPO_WAQ_Distill
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCriticDreamWaQDepth,
    ActorCriticDreamWaQDepthLora,
)
from .log_utils import add_timing_info
from .on_policy_runner import OnPolicyRunner


class DreamWaQDepthDistillRunner(OnPolicyRunner):
    """Student-controlled pure imitation from multiple depth-LoRA teachers."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg,
        log_dir=None,
        device="cpu",
    ):
        self.distillation_cfg = env.cfg.distillation
        super().__init__(env, train_cfg, log_dir, device)

    def _make_policy(self, policy_class, policy_kwargs):
        return policy_class(
            self.env.num_obs,
            self.env.num_actions,
            self.env.num_privileged_obs,
            self.env.num_history_obs,
            self.env.num_latent_dims,
            self.env.num_explicit_dims,
            self.env.num_decoder_output,
            **policy_kwargs,
        ).to(self.device)

    @staticmethod
    def _checkpoint_state(checkpoint):
        return checkpoint.get("model_state_dict", checkpoint)

    def _make_teacher(self, teacher_cfg):
        rank = int(teacher_cfg.get("rank", 8))
        teacher_actor_critic = teacher_cfg.get("teacher_actor_critic", "ActorCriticDreamWaQDepthLora")
        kwargs = dict(self.policy_cfg)
        if "lora" in teacher_actor_critic.lower():
            kwargs.update(
                base_model=teacher_cfg.get("base_model",self.distillation_cfg.base_model),
                actor_ranks= int(teacher_cfg.get("rank",self.distillation_cfg.lora_rank)),
                encoder_ranks=teacher_cfg.get("encoder_ranks", rank),
                decoder_ranks=teacher_cfg.get("decoder_ranks", rank),
                latent_mu_rank=teacher_cfg.get("latent_mu_rank", rank),
                vel_mu_rank=teacher_cfg.get("vel_mu_rank", rank),
                latent_var_ranks=teacher_cfg.get(
                    "latent_var_ranks", rank
                ),
                vel_var_ranks=teacher_cfg.get("vel_var_ranks", rank),
                visual_encoder_ranks=teacher_cfg.get(
                    "visual_encoder_ranks", rank
                ),
            )

        # Reuses the repo's existing LoRA class. Its constructor first loads
        # the configured baseline into the LoRA-wrapped network.
        teacher = self._make_policy(
            eval(teacher_actor_critic),
            kwargs,
        )

        # Then restore the skill-specific saved LoRA checkpoint.
        checkpoint = torch.load(
            teacher_cfg["checkpoint"],
            map_location=self.device,
        )
        teacher.load_state_dict(
            self._checkpoint_state(checkpoint)
        )
        teacher.eval()
        teacher.requires_grad_(False)
        print(
            "Loaded teacher "
            f"{teacher_cfg.get('name', '<unnamed>')}: "
            f"{teacher_cfg['checkpoint']}"
        )
        return teacher

    def _init_agent_and_algo(self):
        # One ordinary generalist depth DreamWaQ student.
        student_class = eval(self.cfg["policy_class_name"])
        student = self._make_policy(
            student_class,
            dict(self.policy_cfg),
        )

        teachers = [
            self._make_teacher(teacher_cfg)
            for teacher_cfg in self.distillation_cfg.teachers
        ]

        algorithm_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_WAQ_Distill = algorithm_class(
            student,
            teachers=teachers,
            device=self.device,
            **self.alg_cfg,
        )

    def _init_storage(self):
        self.alg.init_storage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.num_steps_per_env,
            actor_obs_shape=[self.env.num_obs],
            obs_history_shape=[self.env.num_history_obs],
            depth_image_shape=[1, *self.env.output_resolution],
            action_shape=[self.env.num_actions],
        )

    def learn(
        self,
        num_learning_iterations,
        init_at_random_ep_len=False,
    ):
        self._pre_learn(init_at_random_ep_len)

        if self.current_learning_iteration == 0:
            teacher_ids = self.env.get_teacher_ids()
            unique_ids, counts = torch.unique(teacher_ids,return_counts=True)
            print("Teacher ID check:",teacher_ids.shape,teacher_ids.dtype,unique_ids,counts)

        (
            obs,
            privileged_obs,
            obs_history,
            explicit_info_labels,
            next_state,
            depth_image,
        ) = self.env.get_observations()

        obs = obs.to(self.device)
        obs_history = obs_history.to(self.device)
        depth_image = depth_image.to(self.device)

        self.alg.actor_critic.train()
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, device=self.device
        )

        total_iterations = (
            self.current_learning_iteration
            + num_learning_iterations
        )

        for iteration in range(
            self.current_learning_iteration,
            total_iterations,
        ):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    teacher_ids = self.env.get_teacher_ids().to(
                        self.device
                    )

                    # Student action and teacher labels use state t.
                    actions = self.alg.act(
                        obs,
                        obs_history,
                        depth_image,
                        teacher_ids,
                    )

                    (
                        obs,
                        privileged_obs,
                        obs_history,
                        explicit_info_labels,
                        next_state,
                        rewards,
                        dones,
                        infos,
                        depth_image,
                    ) = self.env.step(actions)

                    obs = obs.to(self.device)
                    obs_history = obs_history.to(self.device)
                    depth_image = depth_image.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Rewards/dones are not part of the imitation loss.
                    self.alg.process_env_step(
                        rewards,
                        dones,
                        infos,
                    )

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        done_ids = (dones > 0).nonzero(
                            as_tuple=False
                        ).flatten()
                        if done_ids.numel():
                            rewbuffer.extend(
                                cur_reward_sum[done_ids]
                                .cpu()
                                .tolist()
                            )
                            lenbuffer.extend(
                                cur_episode_length[done_ids]
                                .cpu()
                                .tolist()
                            )
                            cur_reward_sum[done_ids] = 0
                            cur_episode_length[done_ids] = 0

            collection_time = time.time() - start
            start = time.time()
            mean_loss, stats = self.alg.update()
            learn_time = time.time() - start

            if self.log_dir is not None:
                self._log_distill(
                    iteration,
                    total_iterations,
                    mean_loss,
                    stats,
                    collection_time,
                    learn_time,
                    ep_infos,
                    rewbuffer,
                    lenbuffer,
                )

            if iteration % self.save_interval == 0:
                self.save(
                    os.path.join(
                        self.log_dir,
                        f"model_{iteration}.pt",
                    )
                )
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(
                self.log_dir,
                f"model_{self.current_learning_iteration}.pt",
            )
        )

    def _log_distill(
        self,
        iteration,
        total_iterations,
        mean_loss,
        stats,
        collection_time,
        learn_time,
        ep_infos,
        rewbuffer,
        lenbuffer,
    ):
        iteration_time = collection_time + learn_time
        fps = int(
            self.num_steps_per_env
            * self.env.num_envs
            / max(iteration_time, 1.0e-9)
        )
        self.tot_timesteps += (
            self.num_steps_per_env * self.env.num_envs
        )
        self.tot_time += iteration_time

        self.writer.add_scalar(
            "Loss/distillation", mean_loss, iteration
        )
        self.writer.add_scalar(
            "Loss/action_l1_mean",
            stats["action_l1_mean"],
            iteration,
        )
        self.writer.add_scalar(
            "Loss/action_mse_mean",
            stats["action_mse_mean"],
            iteration,
        )
        self.writer.add_scalar(
            "Loss/learning_rate",
            stats["learning_rate"],
            iteration,
        )

        # Keep the original depth runner's Episode/* namespace so reward
        # components and curriculum statistics remain directly comparable.
        episode_stats = {}
        if ep_infos:
            episode_keys = set().union(
                *(episode_info.keys() for episode_info in ep_infos)
            )
            for key in sorted(episode_keys):
                values = []
                for episode_info in ep_infos:
                    if key not in episode_info:
                        continue
                    value = episode_info[key]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().to(self.device).reshape(-1)
                    else:
                        value = torch.as_tensor(
                            value,
                            dtype=torch.float32,
                            device=self.device,
                        ).reshape(-1)
                    values.append(value)
                if values:
                    mean_value = (
                        torch.cat(values).float().mean().item()
                    )
                    episode_stats[key] = mean_value
                    self.writer.add_scalar(
                        "Episode/" + key,
                        mean_value,
                        iteration,
                    )

        # The student acts deterministically during distillation, but logging
        # its (frozen) policy standard deviation preserves dashboard parity.
        mean_std = self.alg.actor_critic.std.mean().item()
        self.writer.add_scalar(
            "Policy/mean_noise_std", mean_std, iteration
        )
        self.writer.add_scalar("Perf/total_fps", fps, iteration)
        self.writer.add_scalar(
            "Perf/collection time", collection_time, iteration
        )
        self.writer.add_scalar(
            "Perf/learning_time", learn_time, iteration
        )

        mean_reward = None
        mean_length = None
        if rewbuffer:
            mean_reward = statistics.mean(rewbuffer)
            mean_length = statistics.mean(lenbuffer)
            self.writer.add_scalar(
                "Train/mean_reward", mean_reward, iteration
            )
            self.writer.add_scalar(
                "Train/mean_episode_length",
                mean_length,
                iteration,
            )
            self.writer.add_scalar(
                "Train/mean_reward/time",
                mean_reward,
                self.tot_time,
            )
            self.writer.add_scalar(
                "Train/mean_episode_length/time",
                mean_length,
                self.tot_time,
            )
        width = 80
        pad = 35
        episode_lines = "".join(
            f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"
            for key, value in episode_stats.items()
        )
        iteration_header = (
            f" \033[1m Learning iteration "
            f"{iteration}/{total_iterations} \033[0m "
        )
        log_string = (
            f"{'#' * width}\n"
            f"{iteration_header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {collection_time:.3f}s, "
            f"learning {learn_time:.3f}s)\n"
            f"{'Distillation loss:':>{pad}} {mean_loss:.4f}\n"
            f"{'Action L1 mean:':>{pad}} "
            f"{stats['action_l1_mean']:.4f}\n"
            f"{'Action MSE mean:':>{pad}} "
            f"{stats['action_mse_mean']:.4f}\n"
            f"{'Learning rate:':>{pad}} "
            f"{stats['learning_rate']:.6g}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"
        )
        if mean_reward is not None and mean_length is not None:
            log_string += (
                f"{'Mean reward:':>{pad}} {mean_reward:.2f}\n"
                f"{'Mean episode length:':>{pad}} {mean_length:.2f}\n"
            )

        log_string += add_timing_info(
            self.current_learning_iteration,
            total_iterations - self.current_learning_iteration,
            iteration,
            episode_lines,
            iteration_time,
            width,
            pad,
            self.tot_timesteps,
            self.tot_time,
        )
        print(log_string)
