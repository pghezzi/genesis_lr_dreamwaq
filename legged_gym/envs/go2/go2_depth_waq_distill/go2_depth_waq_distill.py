from __future__ import annotations

from legged_gym.envs.go2.go2_depth_waq.go2_depth_waq import Go2DepthWaq
import torch

class Go2DepthWaqDistill(Go2DepthWaq):
    """Existing depth DreamWaQ environment plus teacher-ID exposure."""

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
        
        self.teacher_ids = self._make_teacher_ids()
        self._print_assignment_summary()

    def _print_assignment_summary(self) -> None:
        """Print the realized terrain-column and teacher allocation."""
        teacher_ids = self.teacher_ids.view(-1)
        terrain_types = self.simulator._terrain_types.view(-1)
        teachers = self.cfg.distillation.teachers

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
        if not hasattr(self.simulator, "terrain_types"):
            raise RuntimeError(
                "Teacher IDs require Genesis terrain types. "
                "Use terrain.mesh_type='heightfield' or 'trimesh'."
            )

        terrain_types = self.simulator.terrain_types.to(
            device=self.device,
            dtype=torch.long,
        )
        mapping = getattr(
            self.cfg.distillation,
            "terrain_type_to_teacher",
            None,
        )

        if mapping is None:
            teacher_ids = terrain_types.clone()
        else:
            lookup = torch.as_tensor(
                mapping,
                dtype=torch.long,
                device=self.device,
            )
            required = int(terrain_types.max().item()) + 1
            if lookup.numel() < required:
                raise ValueError(
                    "terrain_type_to_teacher has "
                    f"{lookup.numel()} entries but needs at least {required}."
                )
            teacher_ids = lookup[terrain_types]

        num_teachers = len(self.cfg.distillation.teachers)
        if num_teachers == 0:
            raise ValueError("At least one LoRA teacher must be configured.")
        if torch.any(teacher_ids < 0) or torch.any(
            teacher_ids >= num_teachers
        ):
            raise ValueError(
                f"Teacher IDs must be in range [0, {num_teachers - 1}]."
            )

        # Required shape from the PI: [num_envs, 1].
        return teacher_ids.view(self._num_envs, 1).contiguous()    

    def get_teacher_ids(self) -> torch.Tensor:
        teacher_ids = self.teacher_ids
        expected = (self.num_envs, 1)

        if tuple(teacher_ids.shape) != expected:
            raise RuntimeError(
                f"Expected teacher_ids shape {expected}; "
                f"got {tuple(teacher_ids.shape)}."
            )

        return teacher_ids