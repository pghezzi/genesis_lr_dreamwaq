# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStoragePos:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.observation_history = None
            self.dones = None

            self.torso_velo_targets = None  # same timestep as observations, used by encoder output
            self.grf_targets = None  # next time-step from observations, used by decoder output
            self.obs_targets = None  # next time-step from observations, used by decoder output

            self.pos_actions = None
            self.pos_rewards = None
            self.pos_values = None
            self.pos_actions_log_prob = None
            self.pos_action_mean = None
            self.pos_action_sigma = None
            
            self.hidden_states = None
        
        def clear(self):
            self.__init__()

    # We want all of the actions and associated data formatted in the Model kinematic definition - [FR, FL, RR, RL]
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, obs_hist_shape, actions_shape, torso_velo_shape, grf_shape, device="cpu"):

        self.device = device

        self.first_mean = 0.0
        self.second_mean = 0.0

        self.obs_shape = obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.observation_history = torch.zeros(num_transitions_per_env, num_envs, *obs_hist_shape, device=self.device)    
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        
        # specific to DreamWaQ style history encoder...
        self.torso_velo_targets = torch.zeros(num_transitions_per_env, num_envs, *torso_velo_shape, device=self.device)
        self.grf_targets = torch.zeros(num_transitions_per_env, num_envs, *grf_shape, device=self.device)
        self.observation_targets = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        
        # For PPO
        # Need a set of these for each "task" (position control and torque control)
        self.pos_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.pos_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.pos_actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.pos_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.pos_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.pos_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.pos_mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.pos_sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        #  Shared
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs


        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        
        # Specific to DreamWaQ style history encoder
        self.torso_velo_targets[self.step].copy_(transition.torso_velo_targets)
        self.grf_targets[self.step].copy_(transition.grf_targets)
        self.observation_targets[self.step].copy_(transition.obs_targets)
        
        # Need a set for each "task"
        #  - Position Control
        self.pos_actions[self.step].copy_(transition.pos_actions)
        self.pos_rewards[self.step].copy_(transition.pos_rewards.view(-1, 1))
        self.pos_values[self.step].copy_(transition.pos_values)
        self.pos_actions_log_prob[self.step].copy_(transition.pos_actions_log_prob.view(-1, 1))
        self.pos_mu[self.step].copy_(transition.pos_action_mean)
        self.pos_sigma[self.step].copy_(transition.pos_action_sigma)
        
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns_pos(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.pos_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.pos_rewards[step] + next_is_not_terminal * gamma * next_values - self.pos_values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.pos_returns[step] = advantage + self.pos_values[step]

        # Compute and normalize the advantages
        self.pos_advantages = self.pos_returns - self.pos_values
        self.pos_advantages = (self.pos_advantages - self.pos_advantages.mean()) / (self.pos_advantages.std() + 1e-8)
    
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        critic_observations = self.critic_observations.flatten(0, 1)
        obs_history = self.observation_history.flatten(0,1)

        torso_velo_labels = self.torso_velo_targets.flatten(0,1)
        grf_labels = self.grf_targets.flatten(0,1)
        obs_targets = self.observation_targets.flatten(0,1)

        pos_actions = self.pos_actions.flatten(0, 1)
        pos_values = self.pos_values.flatten(0, 1)
        pos_returns = self.pos_returns.flatten(0, 1)
        pos_old_actions_log_prob = self.pos_actions_log_prob.flatten(0, 1)
        pos_advantages = self.pos_advantages.flatten(0, 1)
        pos_old_mu = self.pos_mu.flatten(0, 1)
        pos_old_sigma = self.pos_sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                # Baseline PPO stuff
                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                obs_hist_batch = obs_history[batch_idx]

                # DreamWaQ Style History Encoder stuff
                torso_velo_labels_batch = torso_velo_labels[batch_idx]
                grf_labels_batch = grf_labels[batch_idx]
                obs_labels_batch = obs_targets[batch_idx]

                # Position Control RL Task
                pos_actions_batch = pos_actions[batch_idx]
                pos_target_values_batch = pos_values[batch_idx]
                pos_returns_batch = pos_returns[batch_idx]
                pos_old_actions_log_prob_batch = pos_old_actions_log_prob[batch_idx]
                pos_advantages_batch = pos_advantages[batch_idx]
                pos_old_mu_batch = pos_old_mu[batch_idx]
                pos_old_sigma_batch = pos_old_sigma[batch_idx]
                
                yield obs_batch, critic_observations_batch, obs_hist_batch, torso_velo_labels_batch, \
                        grf_labels_batch, obs_labels_batch, pos_actions_batch, pos_target_values_batch, \
                        pos_advantages_batch, pos_returns_batch, pos_old_actions_log_prob_batch, pos_old_mu_batch, \
                        pos_old_sigma_batch

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj