from __future__ import annotations

from typing import Dict, Iterator

import torch


class RolloutStorageWAQDistill:
    """Storage for pure action-label imitation.

    The existing DreamWaQ depth storage is PPO-specific. This storage keeps
    only the tensors required to recompute student actions and compare them
    against selected LoRA-teacher actions.
    """

    class Transition:
        def __init__(self):
            self.observations = None
            self.observation_histories = None
            self.depth_images = None
            self.teacher_actions = None
            self.teacher_ids = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        obs_history_shape,
        depth_image_shape,
        action_shape,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.step = 0

        shape_prefix = (
            self.num_transitions_per_env,
            self.num_envs,
        )
        self.observations = torch.zeros(
            *shape_prefix, *obs_shape, device=self.device
        )
        self.observation_histories = torch.zeros(
            *shape_prefix, *obs_history_shape, device=self.device
        )
        self.depth_images = torch.zeros(
            *shape_prefix, *depth_image_shape, device=self.device
        )
        self.teacher_actions = torch.zeros(
            *shape_prefix, *action_shape, device=self.device
        )
        self.teacher_ids = torch.zeros(
            *shape_prefix,
            1,
            dtype=torch.long,
            device=self.device,
        )

    def add_transitions(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError("Distillation rollout buffer overflow.")

        self.observations[self.step].copy_(transition.observations)
        self.observation_histories[self.step].copy_(
            transition.observation_histories
        )
        self.depth_images[self.step].copy_(transition.depth_images)
        self.teacher_actions[self.step].copy_(
            transition.teacher_actions
        )
        self.teacher_ids[self.step].copy_(
            transition.teacher_ids.to(dtype=torch.long)
        )
        self.step += 1

    def mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        batch_size = self.step * self.num_envs
        if batch_size == 0:
            return

        observations = self.observations[: self.step].flatten(0, 1)
        histories = self.observation_histories[: self.step].flatten(0, 1)
        depths = self.depth_images[: self.step].flatten(0, 1)
        labels = self.teacher_actions[: self.step].flatten(0, 1)
        teacher_ids = self.teacher_ids[: self.step].flatten(0, 1)

        mini_batch_size = batch_size // int(num_mini_batches)
        if mini_batch_size == 0:
            raise ValueError(
                "num_mini_batches exceeds the rollout batch size."
            )

        usable_size = mini_batch_size * int(num_mini_batches)
        for _ in range(int(num_epochs)):
            indices = torch.randperm(
                batch_size, device=self.device
            )[:usable_size]
            for mini_batch_index in range(int(num_mini_batches)):
                start = mini_batch_index * mini_batch_size
                end = start + mini_batch_size
                batch_indices = indices[start:end]
                yield {
                    "observations": observations[batch_indices],
                    "observation_histories": histories[batch_indices],
                    "depth_images": depths[batch_indices],
                    "teacher_actions": labels[batch_indices],
                    "teacher_ids": teacher_ids[batch_indices],
                }

    def clear(self):
        self.step = 0