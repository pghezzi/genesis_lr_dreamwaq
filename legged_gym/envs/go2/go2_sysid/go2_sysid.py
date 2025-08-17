import numpy as np

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.gs_utils import *
from .go2_sysid_config import GO2SysIDCfg
import pandas as pd
from tqdm import tqdm


class GO2SysID(LeggedRobot):

    def system_id_in_air(self, env_cfg):
        # load motor_data_file, csv
        motor_data_file = self.cfg.sysid_data.file.format(
            LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        motor_data = pd.read_csv(motor_data_file)
        motor_q = motor_data[["jpos0", "jpos1", "jpos2", "jpos3", "jpos4", "jpos5",
                              "jpos6", "jpos7", "jpos8", "jpos9", "jpos10", "jpos11"]].to_numpy()
        # motor_dq = motor_data[["jvel_0","jvel_1","jvel_2","jvel_3","jvel_4","jvel_5"]].to_numpy()
        motor_q_des = motor_data[["jpos0_des", "jpos1_des", "jpos2_des", "jpos3_des", "jpos4_des", "jpos5_des",
                                  "jpos6_des", "jpos7_des", "jpos8_des", "jpos9_des", "jpos10_des", "jpos11_des"]].to_numpy()
        motor_kp = np.random.uniform(
            self.cfg.sysid_param_range.kp_range[0], self.cfg.sysid_param_range.kp_range[1], (self.num_envs, self.num_actions))
        motor_kd = np.random.uniform(
            self.cfg.sysid_param_range.kd_range[0], self.cfg.sysid_param_range.kd_range[1], (self.num_envs, self.num_actions))

        q_real = torch.from_numpy(motor_q).float().to(self.device)
        qd_real = torch.zeros_like(q_real).to(self.device)
        q_des = torch.from_numpy(motor_q_des).float().to(self.device)
        kp_des = torch.from_numpy(motor_kp).float().to(self.device)
        kd_des = torch.from_numpy(motor_kd).float().to(self.device)

        # sample parameters
        joint_damping_range = self.cfg.sysid_param_range.joint_damping_range
        joint_stiffness_range = self.cfg.sysid_param_range.joint_stiffness_range
        joint_armature_range = self.cfg.sysid_param_range.joint_armature_range
        # set parameters
        sampled_dampings = np.zeros((self.num_envs,))
        sampled_stiffness = np.zeros((self.num_envs,))
        sampled_armatures = np.zeros((self.num_envs,))

        # set joint_friction, joint_damping, joint_armature
        # generate random values for each env, then repeat for each action
        joint_dampings = gs_rand_float(
            joint_damping_range[0], joint_damping_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)
        joint_stiffness = gs_rand_float(
            joint_stiffness_range[0], joint_stiffness_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)
        joint_armature = gs_rand_float(
            joint_armature_range[0], joint_armature_range[1], (self.num_envs,1), device=self.device).repeat(1, self.num_actions)

        # assume all joints have the same damping and friction
        for i in range(self.num_envs):
            sampled_dampings[i] = joint_dampings[i][0]
            sampled_stiffness[i] = joint_stiffness[i][0]
            sampled_armatures[i] = joint_armature[i][0]
        self.robot.set_dofs_damping(
            joint_dampings, self.motors_dof_idx)
        self.robot.set_dofs_stiffness(
            joint_stiffness, self.motors_dof_idx)
        self.robot.set_dofs_armature(
            joint_armature, self.motors_dof_idx)

        # check
        dof_dampings = self.robot.get_dofs_damping()
        dof_stiffness = self.robot.get_dofs_stiffness()
        dof_armatures = self.robot.get_dofs_armature()
        for i in range(self.num_envs):
            assert abs(dof_dampings[i][0] - sampled_dampings[i]) < 1e-4
            assert abs(dof_stiffness[i][0] - sampled_stiffness[i]) < 1e-4
            assert abs(dof_armatures[i][0] - sampled_armatures[i]) < 1e-4
        # generating samples
        metric = 0
        sim_q = []

        # reset
        envs_idx = torch.arange(self.num_envs).to(self.device)
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx] += self.env_origins[envs_idx]
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.dof_pos[envs_idx] = (self.default_dof_pos)
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.p_gains[:] = kp_des[:]
        self.d_gains[:] = kd_des[:]

        delay_steps = self.cfg.domain_rand.delay_steps

        for i in tqdm(range(1 + delay_steps, q_real.shape[0])):
            # apply action
            actions = ((q_des[i-1-delay_steps] - self.default_dof_pos) /
                       self.cfg.control.action_scale).tile((self.num_envs, 1))
            # step physics and render each frame
            self.torques = self._compute_torques(actions)
            self.robot.control_dofs_force(self.torques, self.motors_dof_idx)
            self.scene.step()
            self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
            self.dof_vel[:] = self.robot.get_dofs_velocity(
                self.motors_dof_idx)
            # when sampling
            metric = metric + \
                torch.norm(self.dof_pos - q_real[i].unsqueeze(dim=0), dim=-1)
            sim_q.append(self.dof_pos.cpu().numpy())

        metric = metric.detach().cpu().numpy()
        # print("Average metric", np.mean(metric))
        print("best")
        print("damping", sampled_dampings[np.argmin(metric)], "\n",
              "stiffness", sampled_stiffness[np.argmin(metric)], "\n",
              "armature", sampled_armatures[np.argmin(metric)], "\n",
              #   "limb_mass_ratios", self.sampled_link_mass_scales[np.argmin(metric)], "\n",
              #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[self.feet_indices[0]].friction,
              #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].restitution,
              #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[i].mass for i in range(self.num_bodies)],
              #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].com,
              "kp", kp_des[np.argmin(metric)], "\n",
              "kd", kd_des[np.argmin(metric)], "\n",
              "metric", metric[np.argmin(metric)])
        # print("worst", "damping", sampled_dampings[np.argmax(metric)],
        #       "friction", sampled_frictions[np.argmax(metric)],
        #       "limb_mass_ratios", self.sampled_limb_mass_scales[np.argmax(metric)],
        #     #   "feet_friction", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[self.feet_indices[0]].friction,
        #     #   "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].restitution,
        #     #   "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[i].mass for i in range(self.num_bodies)],
        #     #   "com", self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].com,
        #        "kp", kp_des[np.argmax(metric)],
        #        "kd", kd_des[np.argmax(metric)],
        #         "armature", sampled_armatures[np.argmax(metric)],
        #        metric[np.argmax(metric)])

    def _reset_dofs(self, envs_idx):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """

        self.dof_pos[envs_idx, :] = self.default_dof_pos

        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.robot.zero_all_dofs_velocity(envs_idx)
