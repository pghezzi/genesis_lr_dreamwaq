import sys
import numpy as np
import torch
import time
import genesis as gs

from legged_gym.envs.base.base_task import BaseTask

# Base class for RL tasks
class BaseTaskDWQ(BaseTask):

    def __init__(self, cfg, sim_device, headless):
        super.__init__(cfg, sim_device, headless)
        self.num_obs_hist = cfg.env.num_obs_hist
        self.obs_hist_buf = torch.zeros(self.num_envs, self.num_obs_hist*self.num_obs, device = self.device, dtype=gs.tc_float)
        self.prev_privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        self.disturbance_force = torch.zeros((self.num_envs,2),device=self.device,dtype=torch.float)

    def get_observations(self):
        return self.obs_buf, self.obs_hist_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf, self.prev_privileged_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        obs_hist = torch.zeros(self.num_envs, self.num_obs_hist*self.num_obs, device = self.device, dtype=torch.float)
        prev_privileged_obs = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        return obs, privileged_obs, prev_privileged_obs, obs_hist