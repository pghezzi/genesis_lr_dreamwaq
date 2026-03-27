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
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticCTS
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.storage.replay_buffer import ReplayBuffer
import itertools

'''
PPO with concurrent teacher-student architecture, refer to https://clearlab-sustech.github.io/concurrentTS/, 
combined with AMP for motion imitation. 
'''


class PPO_CTS_AMP(PPO):
    actor_critic: ActorCriticCTS

    def __init__(self,
                 actor_critic,
                 discriminator,
                 amp_data,
                 amp_normalizer,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 use_spo=False,
                 device='cpu',
                 encoder_lr=1e-3,      # learning rate for history encoder
                 num_encoder_epochs=1, # number of epochs for history encoder via supervised learning
                 num_teacher=1,
                 amp_replay_buffer_size=100000,
                 disc_lr=1e-4
                 ):

        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device,
        )

        self.encoder_lr = encoder_lr
        self.num_encoder_epochs = num_encoder_epochs
        self.num_teacher = num_teacher
        
        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorageCTS.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device)
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.rl_params = list(self.actor_critic.actor.parameters()) + \
                        list(self.actor_critic.critic.parameters()) + \
                        list(self.actor_critic.privilege_encoder.parameters()) + \
                        [self.actor_critic.std]
        self.optimizer = optim.Adam(self.rl_params, lr=learning_rate)  # do not consider paramters of student encoder during RL update
        self.history_encoder_optimizer = optim.Adam(
            self.actor_critic.history_encoder.parameters(), lr=encoder_lr)    # for history encoder supervised learning update
        disc_params = [
            {'params': self.discriminator.trunk.parameters(),
             'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(),
             'weight_decay': 10e-2, 'name': 'amp_head'}]
        self.disc_optimizer = optim.Adam(disc_params, lr=disc_lr)
        
        self.transition = RolloutStorageCTS.Transition()

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, 
                     privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorageCTS(
            num_envs, self.num_teacher, num_transitions_per_env, actor_obs_shape, 
            privileged_obs_shape, obs_history_shape, critic_obs_shape, action_shape, self.device)

    def act(self, obs, privileged_obs, obs_history, critic_obs, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # In storage, first num_teacher indices are teacher envs, the rest are student envs
        # Compute the actions and values
        teacher_actions = self.actor_critic.act(obs[:self.num_teacher], None, privileged_obs[:self.num_teacher], act_type='teacher').detach()
        teacher_actions_log_prob = self.actor_critic.get_actions_log_prob(teacher_actions).detach()
        teacher_action_mean = self.actor_critic.action_mean.detach()
        teacher_action_sigma = self.actor_critic.action_std.detach()
        student_actions = self.actor_critic.act(obs[self.num_teacher:], obs_history[self.num_teacher:], None, act_type='student').detach()
        student_actions_log_prob = self.actor_critic.get_actions_log_prob(student_actions).detach()
        student_action_mean = self.actor_critic.action_mean.detach()
        student_action_sigma = self.actor_critic.action_std.detach()
        # store the actions and log probs
        self.transition.actions = torch.cat((teacher_actions, student_actions), dim=0)
        self.transition.actions_log_prob = torch.cat((teacher_actions_log_prob, student_actions_log_prob), dim=0)
        self.transition.action_mean = torch.cat((teacher_action_mean, student_action_mean), dim=0)
        self.transition.action_sigma = torch.cat((teacher_action_sigma, student_action_sigma), dim=0)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = privileged_obs.detach()
        self.transition.observation_histories = obs_history.detach()
        self.transition.critic_observations = critic_obs.detach()
        self.transition.values = self.actor_critic.evaluate(
            self.transition.critic_observations).detach()
        # record amp_obs for discriminator training
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        mean_value_loss = 0
        mean_teacher_surrogate_loss = 0
        mean_student_surrogate_loss = 0
        mean_reconstruction_loss = 0
        # amp metrics
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        generator = self._get_data_generator()
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
                self.num_mini_batches)
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            
            teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch, \
            teacher_old_actions_log_prob_batch, teacher_advantages_batch, teacher_old_mu_batch, teacher_old_sigma_batch, \
            student_obs_batch, student_privileged_obs_batch, student_obs_histories_batch, student_actions_batch, \
            student_old_actions_log_prob_batch, student_advantages_batch, \
            critic_obs_batch, target_values_batch, returns_batch, hid_states_batch, masks_batch = sample
            
            # Discriminator loss.
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            disc_loss, amp_loss, grad_pen_loss, policy_d, expert_d = self._compute_amp_loss(
                policy_state, policy_next_state, expert_state, expert_next_state)
            # Update discriminator.
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.disc_optimizer.step()
            
            loss, teacher_surrogate_loss, student_surrogate_loss, value_loss = self._compute_rl_loss(
                teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch, teacher_old_actions_log_prob_batch,
                teacher_advantages_batch, teacher_old_mu_batch, teacher_old_sigma_batch, student_obs_batch, 
                student_obs_histories_batch, student_actions_batch, student_old_actions_log_prob_batch, student_advantages_batch,
                critic_obs_batch, target_values_batch, returns_batch, hid_states_batch, masks_batch)

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.rl_params, self.max_grad_norm)
            self.optimizer.step()
            
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state.cpu().numpy())
                self.amp_normalizer.update(expert_state.cpu().numpy())
            
            mean_value_loss += value_loss.item()
            mean_teacher_surrogate_loss += teacher_surrogate_loss.item()
            mean_student_surrogate_loss += student_surrogate_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()
        
        generator = self._get_data_generator()
        for teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch, \
            teacher_old_actions_log_prob_batch, teacher_advantages_batch, teacher_old_mu_batch, teacher_old_sigma_batch, \
            student_obs_batch, student_privileged_obs_batch, student_obs_histories_batch, student_actions_batch, \
            student_old_actions_log_prob_batch, student_advantages_batch, \
            critic_obs_batch, target_values_batch, returns_batch, hid_states_batch, masks_batch in generator:
            
            # Reconstruction gradient step
            for _ in range(self.num_encoder_epochs):
                reconstruction_loss = self._compute_encoder_loss(student_obs_histories_batch, student_privileged_obs_batch)
                self.history_encoder_optimizer.zero_grad()
                reconstruction_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                self.history_encoder_optimizer.step()
            
                mean_reconstruction_loss += reconstruction_loss.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_teacher_surrogate_loss /= num_updates
        mean_student_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        mean_reconstruction_loss /= (num_updates * self.num_encoder_epochs)
        self.storage.clear()

        return mean_value_loss, mean_teacher_surrogate_loss, mean_student_surrogate_loss, \
            mean_reconstruction_loss, mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred

    def _compute_rl_loss(self, teacher_obs_batch, teacher_privileged_obs_batch, teacher_actions_batch, teacher_old_actions_log_prob_batch,
                         teacher_advantages_batch, teacher_old_mu_batch, teacher_old_sigma_batch, student_obs_batch, 
                         student_obs_histories_batch, student_actions_batch, student_old_actions_log_prob_batch, 
                         student_advantages_batch, critic_obs_batch, target_values_batch, returns_batch, hid_states_batch, masks_batch):
        # Teacher update
        self.actor_critic.act(
                teacher_obs_batch, None, teacher_privileged_obs_batch, act_type='teacher', masks=masks_batch, hidden_states=hid_states_batch[0])
        teacher_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                teacher_actions_batch)
        teacher_entropy_batch = self.actor_critic.entropy
        teacher_mu_batch = self.actor_critic.action_mean
        teacher_sigma_batch = self.actor_critic.action_std
            
        ## Teacher KL, adapt learning rate
        if self.desired_kl != None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(teacher_sigma_batch / teacher_old_sigma_batch + 1.e-5) +
                    (torch.square(teacher_old_sigma_batch) + torch.square(teacher_old_mu_batch - teacher_mu_batch)) /
                    (2.0 * torch.square(teacher_sigma_batch)) - 0.5, axis=-1)
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(
                            1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(
                            1e-2, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

        ## Surrogate loss
        ratio = torch.exp(teacher_actions_log_prob_batch -
                          torch.squeeze(teacher_old_actions_log_prob_batch))
        surrogate = -torch.squeeze(teacher_advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(teacher_advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        teacher_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
        # Student update
        self.actor_critic.act(
            student_obs_batch, student_obs_histories_batch, None, act_type='student', masks=masks_batch, hidden_states=hid_states_batch[0])
        student_actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
            student_actions_batch)
        student_entropy_batch = self.actor_critic.entropy
            
        ## Surrogate loss
        ratio = torch.exp(student_actions_log_prob_batch -
                              torch.squeeze(student_old_actions_log_prob_batch))
        surrogate = -torch.squeeze(student_advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(student_advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                               1.0 + self.clip_param)
        student_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            
        value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
        # Value function loss
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(
                value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()

        total_entropy_batch = torch.cat((teacher_entropy_batch, student_entropy_batch), dim=0)
            
        loss = self.value_loss_coef * value_loss + \
            teacher_surrogate_loss + student_surrogate_loss \
                - self.entropy_coef * (total_entropy_batch.mean())
        
        return loss, teacher_surrogate_loss, student_surrogate_loss, value_loss
    
    def _compute_encoder_loss(self, student_obs_histories_batch, student_privileged_obs_batch):
        encoder_predictions = self.actor_critic.history_encoder(student_obs_histories_batch)
                
        with torch.no_grad(): # don't backpropagate through the encoder targets
            encoder_targets = self.actor_critic.privilege_encoder(student_privileged_obs_batch)

        reconstruction_loss = nn.functional.mse_loss( # use mse loss
            encoder_predictions, encoder_targets)
        
        return reconstruction_loss
    
    def _compute_amp_loss(self, policy_state, policy_next_state,
                                expert_state, expert_next_state):
        if self.amp_normalizer is not None:
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        expert_loss = torch.nn.MSELoss()(
            expert_d, torch.ones(expert_d.size(), device=self.device))
        policy_loss = torch.nn.MSELoss()(
            policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=10)
        disc_loss = amp_loss + grad_pen_loss
        
        return disc_loss, amp_loss, grad_pen_loss, policy_d, expert_d