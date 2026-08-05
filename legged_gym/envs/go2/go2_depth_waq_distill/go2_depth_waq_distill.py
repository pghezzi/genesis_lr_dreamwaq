from __future__ import annotations

import torch

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq import Go2DepthWaq
from legged_gym.simulator.genesis_simulator_distill import (
    GenesisSimulatorDistill,
)


class Go2DepthWaqDistill(Go2DepthWaq):
    """Existing depth DreamWaQ environment plus teacher-ID exposure."""

    simulator_class = GenesisSimulatorDistill

    def __init__(
        self,
        cfg,
        sim_params,
        sim_device,
        headless,
    ):
        super().__init__(
            cfg,
            sim_params,
            sim_device,
            headless,
        )

        if not hasattr(self, "heading"):
            self.heading = torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
            )

    def get_teacher_ids(self) -> torch.Tensor:
        teacher_ids = self.simulator.teacher_ids
        expected = (self.num_envs, 1)

        if tuple(teacher_ids.shape) != expected:
            raise RuntimeError(
                f"Expected teacher_ids shape {expected}; "
                f"got {tuple(teacher_ids.shape)}."
            )

        return teacher_ids