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
        self._print_assignment_summary()

    def _print_assignment_summary(self) -> None:
        """Print the realized terrain-column and teacher allocation."""
        teacher_ids = self._teacher_ids.view(-1)
        terrain_types = self._terrain_types.view(-1)
        teachers = self._cfg.distillation.teachers

        print("Distillation assignment summary:")
        for column in torch.unique(terrain_types, sorted=True).tolist():
            mask = terrain_types == column
            column_teacher_ids = torch.unique(teacher_ids[mask])
            teacher_labels = []
            for teacher_id in column_teacher_ids.tolist():
                name = teachers[teacher_id].get(
                    "name", f"teacher_{teacher_id}"
                )
                teacher_labels.append(f"{teacher_id} ({name})")
            print(
                f"  terrain column {column}: {int(mask.sum().item())} "
                f"env(s) -> teacher {', '.join(teacher_labels)}"
            )

        teacher_counts = torch.bincount(
            teacher_ids,
            minlength=len(teachers),
        )
        for teacher_id, teacher_cfg in enumerate(teachers):
            count = int(teacher_counts[teacher_id].item())
            percentage = 100.0 * count / max(self._num_envs, 1)
            name = teacher_cfg.get("name", f"teacher_{teacher_id}")
            print(
                f"  teacher {teacher_id} ({name}) total: {count} "
                f"env(s), {percentage:.1f}%"
            )

        spread = int(
            (teacher_counts.max() - teacher_counts.min()).item()
        )
        balance = "PASS" if spread <= 1 else "WARNING"
        print(
            f"  teacher balance check: {balance} "
            f"(max count difference {spread})"
        )

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
