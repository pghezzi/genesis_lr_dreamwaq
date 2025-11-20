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
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic_Dynamic, ContextDecoder
from rsl_rl.storage import RolloutStorageDynamics

from .pc_grad import PCGrad
from .zclip import ZClip

class PPODynamic:
    actor_critic: ActorCritic_Dynamic
    decoder_network: ContextDecoder
    def __init__(self,
                 actor_critic,
                 decoder_network,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.pos_learning_rate = learning_rate
        self.tau_learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.act_optimizer, self.enc_optimizer = actor_critic.configure_optimizers(learning_rate)
        self.transition = RolloutStorageDynamics.Transition()
        self.act_optimizer = PCGrad(self.act_optimizer)
        self.enc_optimizer = PCGrad(self.enc_optimizer)

        self.decoder = decoder_network
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        # Initialize ZClip
        self.zclip = ZClip(alpha=0.97, z_thresh=2.5)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, action_shape, torso_velo_shape, grf_shape):
        self.storage = RolloutStorageDynamics(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, \
                                              action_shape, torso_velo_shape, grf_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    # TODO - Try to prevent any crashes by replacing nans and other bad values....
    def act(self, obs, critic_obs, obs_history, torso_velo):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        all_actions = self.actor_critic.act(obs,obs_history).detach()
        # Compute the actions and values
        #  - Position Control
        self.transition.pos_actions =  all_actions[:,0:12]
        self.transition.pos_values = self.actor_critic.evaluate_pos(critic_obs).detach()
        self.transition.pos_actions_log_prob = self.actor_critic.get_pos_actions_log_prob(self.transition.pos_actions).detach()
        self.transition.pos_action_mean = self.actor_critic.pos_action_mean.detach()
        self.transition.pos_action_sigma = self.actor_critic.pos_action_std.detach()
        #  - Torque Control
        self.transition.tau_actions =  all_actions[:,12:24]
        self.transition.tau_values = self.actor_critic.evaluate_tau(critic_obs).detach()
        self.transition.tau_actions_log_prob = self.actor_critic.get_tau_actions_log_prob(self.transition.tau_actions).detach()
        self.transition.tau_action_mean = self.actor_critic.tau_action_mean.detach()
        self.transition.tau_action_sigma = self.actor_critic.tau_action_std.detach()
        
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs
        
        # The current torso velocities, used as a target for part of the encoders output
        self.transition.torso_velo_targets = torso_velo
        
        return all_actions
    
    def process_env_step(self, pos_rewards, tau_rewards, dones, infos, grf_labels, obs_labels):
        self.transition.pos_rewards = pos_rewards.clone()
        self.transition.tau_rewards = tau_rewards.clone()
        
        self.transition.dones = dones
        # Values from the next-time step used as labels for the decoder network
        self.transition.grf_targets = grf_labels
        self.transition.obs_targets = obs_labels
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.pos_rewards += self.gamma * torch.squeeze(self.transition.pos_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.tau_rewards += self.gamma * torch.squeeze(self.transition.tau_values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values_pos = self.actor_critic.evaluate_pos(last_critic_obs).detach()
        self.storage.compute_returns_pos(last_values_pos, self.gamma, self.lam)

        last_values_tau = self.actor_critic.evaluate_tau(last_critic_obs).detach()
        self.storage.compute_returns_tau(last_values_tau, self.gamma, self.lam)

    def update(self,beta=1):
        mean_pos_value_loss = 0
        mean_pos_surrogate_loss = 0
        mean_tau_value_loss = 0
        mean_tau_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_decoder_loss = 0

        # if self.actor_critic.is_recurrent:
        #     generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # else:
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, obs_hist_batch, vel_target, \
            grf_target, obs_target, pos_actions_batch, pos_target_values_batch, \
            pos_advantages_batch, pos_returns_batch, pos_old_actions_log_prob_batch, pos_old_mu_batch, \
            pos_old_sigma_batch,  tau_actions_batch, tau_target_values_batch, \
            tau_advantages_batch, tau_returns_batch, tau_old_actions_log_prob_batch, tau_old_mu_batch, \
            tau_old_sigma_batch, hid_states_batch, masks_batch in generator:

                self.actor_critic.train()
                self.optimizer.zero_grad()
                # PPO stuff
                self.actor_critic.act(obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                # Encoder stuff
                # pull out some values from the actor that I want to use in the decoder...
                #    avoids a second separate run through the encoder + aligns RL update with enc update...
                mean_latent = self.actor_critic.cenet_mean
                logvar_latent = self.actor_critic.cenet_logvar
                cenet_latent = self.actor_critic.cenet_z
                cenet_torso_velo = self.actor_critic.cenet_torso_velo
                # PPO stuff
                #    - Position Control
                pos_actions_log_prob_batch = self.actor_critic.get_pos_actions_log_prob(pos_actions_batch)
                pos_value_batch            = self.actor_critic.evaluate_pos(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                pos_mu_batch               = self.actor_critic.pos_action_mean
                pos_sigma_batch            = self.actor_critic.pos_action_std
                pos_entropy_batch          = self.actor_critic.pos_entropy
                #    - Torque Control
                tau_actions_log_prob_batch = self.actor_critic.get_tau_actions_log_prob(tau_actions_batch)
                tau_value_batch            = self.actor_critic.evaluate_tau(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                tau_mu_batch               = self.actor_critic.tau_action_mean
                tau_sigma_batch            = self.actor_critic.tau_action_std
                tau_entropy_batch          = self.actor_critic.tau_entropy

                # First do the VAE loss function that is shared between both control modailities
                #     Get the prediction from the decoder
                self.decoder.eval()
                dec_input = torch.cat((cenet_latent, cenet_torso_velo), dim=-1)
                enc_update_obs_decode = self.decoder(dec_input)
                
                grf_target.requires_grad = False
                obs_target.requires_grad = False
                
                # decode_target = torch.cat((obs_target, grf_target), dim=-1)
                decode_target = obs_target
                vel_target.requires_grad = False
                


                with torch.no_grad():
                    mean_pred = torch.mean(decode_target, dim=0)
                    mean_pred_error = F.mse_loss(mean_pred, decode_target)
                    actual_pred_error = F.mse_loss(enc_update_obs_decode.clone().detach(),decode_target)
                    ratio = mean_pred_error / actual_pred_error
                    pboot = 1 - torch.tanh(ratio)
                    print("Mean Pred Error: ", mean_pred_error)
                    print("Prediction Error: ", actual_pred_error)
                    print("Raio: ", ratio)
                    print("boot probability: ", pboot)
                    print()




                # autoenc_loss = (nn.MSELoss()(cenet_torso_velo,vel_target) + nn.MSELoss()(enc_update_obs_decode,decode_target) + beta*(-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())))/self.num_mini_batches
                autoenc_loss = F.mse_loss(cenet_torso_velo,vel_target) + F.mse_loss(enc_update_obs_decode,decode_target) + beta*(-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp()))

                # Now calculate the PPO/SPO losses for each RL task
                #   - Position Control
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        pos_kl = torch.sum(
                            torch.log(pos_sigma_batch / pos_old_sigma_batch + 1.e-5) + (torch.square(pos_old_sigma_batch) + torch.square(pos_old_mu_batch - pos_mu_batch)) / (2.0 * torch.square(pos_sigma_batch)) - 0.5, axis=-1)
                        pos_kl_mean = torch.mean(pos_kl)

                        if pos_kl_mean > self.desired_kl * 2.0:
                            self.pos_learning_rate = max(1e-5, self.pos_learning_rate / 1.5)
                        elif pos_kl_mean < self.desired_kl / 2.0 and pos_kl_mean > 0.0:
                            self.pos_learning_rate = min(1e-2, self.pos_learning_rate * 1.5)
                        
                        for param_group in self.optimizer.optimizer.param_groups:
                            # specifically modifies the learning rate of the position-control specific parameters
                            if "name" in param_group.keys():
                                if "pos_branch" in param_group["name"]:
                                    param_group['lr'] = self.pos_learning_rate

                # PPO stuff
                # PPO Surrogate loss
                pos_ratio = torch.exp(pos_actions_log_prob_batch - torch.squeeze(pos_old_actions_log_prob_batch))
                # surrogate = -torch.squeeze(pos_advantages_batch) * pos_ratio
                # surrogate_clipped = -torch.squeeze(pos_advantages_batch) * torch.clamp(pos_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                # surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # SPO Surrogate loss
                pos_surrogate_loss = -(torch.squeeze(pos_advantages_batch) * pos_ratio - torch.abs(torch.squeeze(pos_advantages_batch)) * torch.pow(pos_ratio - 1, 2) / (2 * 0.2)).mean()

                # PPO stuff
                # Value function loss
                if self.use_clipped_value_loss:
                    pos_value_clipped = pos_target_values_batch + (pos_value_batch - pos_target_values_batch).clamp(-self.clip_param, self.clip_param)
                    pos_value_losses = (pos_value_batch - pos_returns_batch).pow(2)
                    pos_value_losses_clipped = (pos_value_clipped - pos_returns_batch).pow(2)
                    pos_value_loss = torch.max(pos_value_losses, pos_value_losses_clipped).mean()
                else:
                    pos_value_loss = (pos_returns_batch - pos_value_batch).pow(2).mean()

                pos_loss = pos_surrogate_loss + self.value_loss_coef * pos_value_loss - self.entropy_coef * pos_entropy_batch.mean()

                #   - Torque Control
                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        tau_kl = torch.sum(
                            torch.log(tau_sigma_batch / tau_old_sigma_batch + 1.e-5) + (torch.square(tau_old_sigma_batch) + torch.square(tau_old_mu_batch - tau_mu_batch)) / (2.0 * torch.square(tau_sigma_batch)) - 0.5, axis=-1)
                        tau_kl_mean = torch.mean(tau_kl)

                        if tau_kl_mean > self.desired_kl * 2.0:
                            self.tau_learning_rate = max(1e-5, self.tau_learning_rate / 1.5)
                        elif tau_kl_mean < self.desired_kl / 2.0 and tau_kl_mean > 0.0:
                            self.tau_learning_rate = min(1e-2, self.tau_learning_rate * 1.5)
                        
                        for param_group in self.optimizer.optimizer.param_groups:
                            # Specifically modifies the parameters specific to the torque-control actions
                            if "name" in param_group.keys():
                                if "tau_branch" in param_group["name"]:
                                    param_group['lr'] = self.tau_learning_rate


                # Surrogate loss
                tau_ratio = torch.exp(tau_actions_log_prob_batch - torch.squeeze(tau_old_actions_log_prob_batch))
                # surrogate = -torch.squeeze(tau_advantages_batch) * tau_ratio
                # surrogate_clipped = -torch.squeeze(tau_advantages_batch) * torch.clamp(tau_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                # tau_surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # SPO Surrogate loss
                tau_surrogate_loss = -(torch.squeeze(tau_advantages_batch) * tau_ratio - torch.abs(torch.squeeze(tau_advantages_batch)) * torch.pow(tau_ratio - 1, 2) / (2 * 0.2)).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    tau_value_clipped = tau_target_values_batch + (tau_value_batch - tau_target_values_batch).clamp(-self.clip_param, self.clip_param)
                    tau_value_losses = (tau_value_batch - tau_returns_batch).pow(2)
                    tau_value_losses_clipped = (tau_value_clipped - tau_returns_batch).pow(2)
                    tau_value_loss = torch.max(tau_value_losses, tau_value_losses_clipped).mean()
                else:
                    tau_value_loss = (tau_returns_batch - tau_value_batch).pow(2).mean()

                tau_loss = tau_surrogate_loss + self.value_loss_coef * tau_value_loss - self.entropy_coef * tau_entropy_batch.mean()

                losses = [pos_loss, tau_loss, autoenc_loss]
                
                # Gradient step
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                # self.optimizer.step()
                
                # PCGrad - back-propigate the loss
                self.optimizer.pc_backward(losses)
                # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.zclip.step(self.actor_critic)
                self.optimizer.step()

                # Perfrom a separate update on the decoder....
                self.decoder.train()
                self.actor_critic.eval()
                self.decoder_optimizer.zero_grad()
                
                dec_out = self.decoder(dec_input.detach())
                dec_loss = F.mse_loss(dec_out, decode_target)
                dec_loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
                self.decoder_optimizer.step()

                mean_pos_value_loss += pos_value_loss.item()
                mean_pos_surrogate_loss += pos_surrogate_loss.item()
                mean_tau_value_loss += tau_value_loss.item()
                mean_tau_surrogate_loss += tau_surrogate_loss.item()
                mean_autoenc_loss += autoenc_loss.item()
                mean_decoder_loss += dec_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_pos_value_loss /= num_updates
        mean_pos_surrogate_loss /= num_updates
        mean_tau_value_loss /= num_updates
        mean_tau_surrogate_loss /= num_updates
        mean_autoenc_loss /= num_updates
        mean_decoder_loss /= num_updates

        self.storage.clear()

        return mean_pos_value_loss, mean_pos_surrogate_loss, mean_tau_value_loss, mean_tau_surrogate_loss, mean_autoenc_loss, mean_decoder_loss
