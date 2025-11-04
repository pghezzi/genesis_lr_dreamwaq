import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
from genesis.engine.solvers.avatar_solver import AvatarSolver
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.dreamwaq import LeggedRobotDreamWaq
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_sqrt_float, quat_apply_yaw, get_scale_shift
from genesis.utils import geom as gu
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.gs_utils import *
from .dreamwaq_config import LeggedRobotCfg
from collections import deque


class LeggedRobotDreamWaq(LeggedRobotDreamWaq):
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset, create entity
             2. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=os.path.join(asset_root, asset_file),
                merge_fixed_links= True,  # if merge_fixed_links is True, then one link may have multiple geometries, which will cause error in set_friction_ratio
                links_to_keep= self.cfg.asset.links_to_keep,
                pos=np.array(self.cfg.init_state.pos),
                quat=np.array(self.cfg.init_state.rot),
                fixed= self.cfg.asset.fix_base_link,
            ),
            visualize_contact=self.debug,
        )

        self.num_dof = len(self.cfg.asset.dof_names)

        # build
        self.scene.build(n_envs=self.num_build_envs)
        self.default_friction = gu.default_friction() #rigid_shape_props_asset[1].friction
        self._get_env_origins()
        self._init_custom_buffers__()

        self._init_domain_params()

        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        # name to indices
        self.motors_dof_idx = [self.robot.get_joint(
            name).dof_start for name in self.cfg.asset.dof_names]

        # find link indices, termination links, penalized links, and feet
        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices
        self.termination_indices = find_link_indices(self.cfg.asset.terminate_after_contacts_on)
        all_link_names = [link.name for link in self.robot.links]
        print(f"all link names: {all_link_names} {self.cfg.asset.terminate_after_contacts_on}")
        print("termination link indices:", self.termination_indices)
        self.penalized_indices = find_link_indices(
            self.cfg.asset.penalize_contacts_on)
        print(f"penalized link indices: {self.penalized_indices}")
        self.feet_names = [
            link.name for link in self.robot.links if self.cfg.asset.foot_name[0] in link.name]
        self.feet_indices = find_link_indices(self.feet_names)
        print(f"feet link indices: {self.feet_indices}")
        assert len(self.termination_indices) > 0
        assert len(self.feet_indices) > 0

        # dof position limits
        self.dof_pos_limits = torch.stack(
            self.robot.get_dofs_limit(self.motors_dof_idx), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motors_dof_idx)[
                                                             1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = (
                m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            )
            self.dof_pos_limits[i, 1] = (
                m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                )


        # randomize friction
        if self.cfg.domain_rand.randomize_friction:
            self._randomize_friction(np.arange(self.num_envs))
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(np.arange(self.num_envs))
        # randomize COM displacement
        if self.cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(np.arange(self.num_envs))
        # randomize joint armature
        if self.cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(np.arange(self.num_envs))
        # randomize joint stiffness
        if self.cfg.domain_rand.randomize_joint_stiffness:
            self._randomize_joint_stiffness(np.arange(self.num_envs))
        # randomize joint damping
        if self.cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(np.arange(self.num_envs))