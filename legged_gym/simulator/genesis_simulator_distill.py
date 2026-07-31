from __future__ import annotations

import torch

from legged_gym.simulator.genesis_simulator import GenesisSimulator


class GenesisSimulatorDistill(GenesisSimulator):
    """Genesis simulator that exposes one teacher/skill ID per environment.

    Genesis already assigns:
      - `_terrain_levels`: terrain difficulty row
      - `_terrain_types`: terrain column/type

    This class does not alter terrain generation. It only maps those existing
    assignments to the teacher IDs used during distillation.
    """

    def __init__(self, cfg, sim_params, device, headless):
        super().__init__(cfg, sim_params, device, headless)
        self._teacher_ids = self._make_teacher_ids()

    def _make_teacher_ids(self) -> torch.Tensor:
        if not hasattr(self, "_terrain_types"):
            raise RuntimeError(
                "Teacher IDs require Genesis terrain types. "
                "Use terrain.mesh_type='heightfield' or 'trimesh'."
            )

        terrain_types = self._terrain_types.to(
            device=self._device,
            dtype=torch.long,
        )
        mapping = getattr(
            self._cfg.distillation,
            "terrain_type_to_teacher",
            None,
        )

        if mapping is None:
            teacher_ids = terrain_types.clone()
        else:
            lookup = torch.as_tensor(
                mapping,
                dtype=torch.long,
                device=self._device,
            )
            required = int(terrain_types.max().item()) + 1
            if lookup.numel() < required:
                raise ValueError(
                    "terrain_type_to_teacher has "
                    f"{lookup.numel()} entries but needs at least {required}."
                )
            teacher_ids = lookup[terrain_types]

        num_teachers = len(self._cfg.distillation.teachers)
        if num_teachers == 0:
            raise ValueError("At least one LoRA teacher must be configured.")
        if torch.any(teacher_ids < 0) or torch.any(
            teacher_ids >= num_teachers
        ):
            raise ValueError(
                f"Teacher IDs must be in [0, {num_teachers - 1}]."
            )

        # Required shape from the PI: [num_envs, 1].
        return teacher_ids.view(self._num_envs, 1).contiguous()

    @property
    def teacher_ids(self) -> torch.Tensor:
        return self._teacher_ids