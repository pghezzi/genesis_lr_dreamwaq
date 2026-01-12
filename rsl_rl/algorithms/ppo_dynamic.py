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
from torch import linalg as LA

import numpy as np
import random

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
                 pinn_lambda=0.001,
                 pinn_warmup=1000,
                 pinn_init_steps=500,
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
        self.act_optimizer = PCGrad(self.act_optimizer, reduction='sum')
        # self.enc_optimizer = PCGrad(self.enc_optimizer, reduction='sum')

        # # We want to reduce the LR of the critic
        for param_group in self.act_optimizer.optimizer.param_groups:
        # for param_group in self.act_optimizer.param_groups:
            # specifically modifies the learning rate of the position-control specific parameters
            if "name" in param_group.keys():
                if "critic" in param_group["name"]:
                    param_group['lr'] = (learning_rate / 3.0)

        self.decoder = decoder_network
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        self.boot_mult = 1.0
        self.use_boot = False

        self.pinn_weight_final = pinn_lambda
        self.pinn_weight = 0.0
        self.pinn_warmup_steps = pinn_warmup
        self.pinn_init = pinn_init_steps

        self.num_pinn_updates = 0

        # # Initialize ZClip
        # self.act_zclip = ZClip(alpha=0.97, z_thresh=2.5)
        # self.enc_zclip = ZClip(alpha=0.97, z_thresh=2.5)

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

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, action_shape, torso_velo_shape, grf_shape, wb_shape):
        self.storage = RolloutStorageDynamics(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, \
                                              action_shape, torso_velo_shape, grf_shape, wb_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, obs_history, torso_velo, prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist):
        # if self.actor_critic.is_recurrent:
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        if self.use_boot:
            all_actions = self.actor_critic.act(obs,obs_history).detach()
        else:
            all_actions = self.actor_critic.act_bootmask(obs,obs_history).detach()

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

        # PINN stuff
        self.transition.prev_obs      = prev_obs
        self.transition.prev_obs_hist = prev_obs_hist
        self.transition.pprev_obs      = pprev_obs
        self.transition.pprev_obs_hist = pprev_obs_hist
        
        return all_actions
    
    def process_env_step(self, pos_rewards, tau_rewards, dones, infos, grf_labels, obs_labels, gt_forces, mass_mats, bias_vecs, torso_acc):
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

        # PINN stuff
        self.transition.wb_contact_forces = gt_forces
        self.transition.wb_mass_mat = mass_mats
        self.transition.wb_bias_vec = bias_vecs
        self.transition.torso_acc = torso_acc
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def spectral_normalization(self, model):
        """Applies spectral normalization to linear and attention layers.
        
        Normalizes weights such that their spectral norm (2-norm) doesn't exceed
        the specified bound. Only affects Linear and MultiheadAttention layers.

        Args:
            model: The neural network model to normalize.

        Note:
            - Operates in-place on the model parameters
            - Only processes weights (not biases)
            - Only affects parameters with ndim > 1
        """
        whitelist = (nn.Linear, nn.MultiheadAttention)

        for module in model.modules():
            if isinstance(module, whitelist):
                for name, param in module.named_parameters():
                    if name.endswith("weight") and param.ndim > 1:
                        with torch.no_grad():
                            weight = param.data
                            norm = LA.matrix_norm(weight, ord=2)
                            
                            # Normalize if exceeds bound
                            if norm > 2.0:
                                param.data = (weight / norm) * 2.0
    
    def compute_returns(self, last_critic_obs):
        last_values_pos = self.actor_critic.evaluate_pos(last_critic_obs).detach()
        self.storage.compute_returns_pos(last_values_pos, self.gamma, self.lam)

        last_values_tau = self.actor_critic.evaluate_tau(last_critic_obs).detach()
        self.storage.compute_returns_tau(last_values_tau, self.gamma, self.lam)

    def update(self, action_func, fb_func, dt, itr, beta=1.0):
        mean_pos_value_loss = 0
        mean_pos_surrogate_loss = 0
        mean_tau_value_loss = 0
        mean_tau_surrogate_loss = 0
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_recon_loss = 0
        mean_kld_loss = 0
        mean_decoder_loss = 0
        mean_pinn_loss = 0

        all_enc_obs_targets = []
        all_enc_recons     = []

        use_pinn = True

        if itr > self.pinn_init and self.num_pinn_updates < (self.pinn_warmup_steps+1):
            self.pinn_weight = (float(self.num_pinn_updates)/float(self.pinn_warmup_steps))*self.pinn_weight_final
            print(self.pinn_weight)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, obs_hist_batch, vel_target, \
            grf_target, obs_target, pos_actions_batch, pos_target_values_batch, \
            pos_advantages_batch, pos_returns_batch, pos_old_actions_log_prob_batch, pos_old_mu_batch, \
            pos_old_sigma_batch,  tau_actions_batch, tau_target_values_batch, \
            tau_advantages_batch, tau_returns_batch, tau_old_actions_log_prob_batch, tau_old_mu_batch, \
            tau_old_sigma_batch, prev_obs_batch, prev_obs_hist_batch, gt_forces_batch, mass_mat_batch, \
            bias_vec_batch, torso_accs_batch,  pprev_obs_batch, pprev_obs_hist_batch  in generator:

                self.actor_critic.train()
                self.act_optimizer.zero_grad()
                self.enc_optimizer.zero_grad()

                if self.use_boot:
                    self.actor_critic.act(obs_batch, obs_hist_batch)
                else:
                    self.actor_critic.act_bootmask(obs_batch, obs_hist_batch)

                current_actions = torch.cat([self.actor_critic.mean_pos, self.actor_critic.mean_tau], dim=-1)
                
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
                pos_value_batch            = self.actor_critic.evaluate_pos(critic_obs_batch)
                pos_mu_batch               = self.actor_critic.pos_action_mean
                pos_sigma_batch            = self.actor_critic.pos_action_std
                pos_entropy_batch          = self.actor_critic.pos_entropy
                #    - Torque Control
                tau_actions_log_prob_batch = self.actor_critic.get_tau_actions_log_prob(tau_actions_batch)
                tau_value_batch            = self.actor_critic.evaluate_tau(critic_obs_batch)
                tau_mu_batch               = self.actor_critic.tau_action_mean
                tau_sigma_batch            = self.actor_critic.tau_action_std
                tau_entropy_batch          = self.actor_critic.tau_entropy

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
                        
                        for param_group in self.act_optimizer.optimizer.param_groups:
                        # for param_group in self.act_optimizer.param_groups:
                            # specifically modifies the learning rate of the position-control specific parameters
                            if "name" in param_group.keys():
                                if "pos_branch" in param_group["name"]:
                                    param_group['lr'] = self.pos_learning_rate

                # PPO stuff
                # PPO Surrogate loss
                pos_ratio = torch.exp(pos_actions_log_prob_batch - torch.squeeze(pos_old_actions_log_prob_batch))
                pos_surrogate = -torch.squeeze(pos_advantages_batch) * pos_ratio
                pos_surrogate_clipped = -torch.squeeze(pos_advantages_batch) * torch.clamp(pos_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                pos_surrogate_loss = torch.max(pos_surrogate, pos_surrogate_clipped).mean()

                # SPO Surrogate loss
                # pos_surrogate_loss = -(torch.squeeze(pos_advantages_batch) * pos_ratio - torch.abs(torch.squeeze(pos_advantages_batch)) * torch.pow(pos_ratio - 1, 2) / (2 * 0.2)).mean()

                # PPO stuff
                # Value function loss
                if self.use_clipped_value_loss:
                    pos_value_clipped = pos_target_values_batch + (pos_value_batch - pos_target_values_batch).clamp(-self.clip_param, self.clip_param)
                    pos_value_losses = (pos_value_batch - pos_returns_batch).pow(2)
                    pos_value_losses_clipped = (pos_value_clipped - pos_returns_batch).pow(2)
                    pos_value_loss = torch.max(pos_value_losses, pos_value_losses_clipped).mean()
                else:
                    pos_value_loss = (pos_returns_batch - pos_value_batch).pow(2).mean()

                pos_loss = pos_surrogate_loss + self.value_loss_coef * pos_value_loss - 0.001 * pos_entropy_batch.mean()

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
                        
                        for param_group in self.act_optimizer.optimizer.param_groups:
                        # for param_group in self.act_optimizer.param_groups:
                            # Specifically modifies the parameters specific to the torque-control actions
                            if "name" in param_group.keys():
                                if "tau_branch" in param_group["name"]:
                                    param_group['lr'] = self.tau_learning_rate


                # Surrogate loss
                tau_ratio = torch.exp(tau_actions_log_prob_batch - torch.squeeze(tau_old_actions_log_prob_batch))
                tau_surrogate = -torch.squeeze(tau_advantages_batch) * tau_ratio
                tau_surrogate_clipped = -torch.squeeze(tau_advantages_batch) * torch.clamp(tau_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                tau_surrogate_loss = torch.max(tau_surrogate, tau_surrogate_clipped).mean()

                # SPO Surrogate loss
                # tau_surrogate_loss = -(torch.squeeze(tau_advantages_batch) * tau_ratio - torch.abs(torch.squeeze(tau_advantages_batch)) * torch.pow(tau_ratio - 1, 2) / (2 * 0.2)).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    tau_value_clipped = tau_target_values_batch + (tau_value_batch - tau_target_values_batch).clamp(-self.clip_param, self.clip_param)
                    tau_value_losses = (tau_value_batch - tau_returns_batch).pow(2)
                    tau_value_losses_clipped = (tau_value_clipped - tau_returns_batch).pow(2)
                    tau_value_loss = torch.max(tau_value_losses, tau_value_losses_clipped).mean()
                else:
                    tau_value_loss = (tau_returns_batch - tau_value_batch).pow(2).mean()

                tau_loss = tau_surrogate_loss + self.value_loss_coef * tau_value_loss - self.entropy_coef * tau_entropy_batch.mean()

                ###
                #  PINN Loss
                ###
                # Use the model to generate the "previous" actions (action-pred using previous obs)
                pinn_loss = 0.0
                task_align_loss = 0.0
                task_ratio_loss = 0.0
                if self.pinn_weight > 0.0:
                    if self.use_boot:
                        self.actor_critic.act(prev_obs_batch, prev_obs_hist_batch)
                    else:
                        self.actor_critic.act_bootmask(prev_obs_batch, prev_obs_hist_batch)
                    prev_actions = torch.cat([self.actor_critic.mean_pos, self.actor_critic.mean_tau], dim=-1)


                    pprev_actions = None
                    if self.use_boot:
                        self.actor_critic.act(pprev_obs_batch, pprev_obs_hist_batch)
                    else:
                        self.actor_critic.act_bootmask(pprev_obs_batch, pprev_obs_hist_batch)
                    pprev_actions = torch.cat([self.actor_critic.mean_pos, self.actor_critic.mean_tau], dim=-1)

                    # Process current and previous actions into the action-space
                    q_des_curr, tau_des_curr   = action_func(current_actions)
                    q_des_prev, tau_des_prev   = action_func(prev_actions)
                    q_des_pprev, tau_des_pprev = action_func(pprev_actions)

                    # Extract joint pose and velocity data
                    # Obs - cmd (3), proj_grav (3), ang_vel (3)
                    q_pos_curr,  q_velo_curr  = obs_batch[:,8:20].detach().clone(),       obs_batch[:,20:32].detach().clone()
                    q_pos_prev,  q_velo_prev  = prev_obs_batch[:,8:20].detach().clone(),  prev_obs_batch[:,20:32].detach().clone()
                    q_pos_pprev, q_velo_pprev = pprev_obs_batch[:,8:20].detach().clone(), pprev_obs_batch[:,20:32].detach().clone()

                    # Calculate feedback torques
                    # fb_func
                    pd_tau_curr  = fb_func(q_des_curr,  q_pos_curr,  q_velo_curr)
                    pd_tau_prev  = fb_func(q_des_prev,  q_pos_prev,  q_velo_prev)
                    pd_tau_pprev = fb_func(q_des_pprev, q_pos_pprev, q_velo_pprev)

                    ###
                    #   WB-dynamics
                    ###
                    # Use 1st order backwards finite differences to approximate models command acceleration
                    dof_acc = (q_des_curr - 2.0*q_des_prev + q_des_pprev) / np.power(dt,2)
                    # Create the whole-body acceleration vector
                    wb_acc = torch.cat([torso_accs_batch, dof_acc], dim=1).float()
                    # Create the whole-boyd tau vector 
                    wb_tau = torch.cat([torch.zeros(torso_accs_batch.shape[0], 6).float().to(self.device), (tau_des_curr.float() + pd_tau_curr.float())], dim=1).float()

                    # Calculate the models wb-dynamics
                    model_wb_dynamics = torch.bmm(mass_mat_batch.float(), wb_acc.unsqueeze(-1)).squeeze(-1) + bias_vec_batch.float()

                    error = model_wb_dynamics[:,6:] - gt_forces_batch[:,6:] - wb_tau[:,6:]

                    # softly weight by contact
                    # Apply soft (to make this a continuous reward signal) contact weighting to avoid over-penalizing for leg movement
                    contact_magnitude = torch.clamp(gt_forces_batch[:,6:], min=0.0)
                    contact_max = torch.max(contact_magnitude, dim=1, keepdim=True)[0]
                    contact_weight = contact_magnitude / (contact_max + 1e-8)
                    error *= contact_weight

                    # Make the error relative, so that it is less senesitive to scale
                    rel_error = torch.norm(error, dim=1) / (1e-8 + torch.norm(wb_tau[:,6:].detach().clone(), dim=1) + torch.norm(gt_forces_batch[:,6:], dim=1))

                    # Calculate the whole-body PINN loss
                    pinn_loss = torch.mean(rel_error)

                    # Calcualte values used by the next two losses
                    ff_power_curr, pd_power_curr   = tau_des_curr*q_velo_curr,   pd_tau_curr*q_velo_curr
                    ff_power_prev, pd_power_prev   = tau_des_prev*q_velo_prev,   pd_tau_prev*q_velo_prev
                    ff_power_pprev, pd_power_pprev = tau_des_pprev*q_velo_pprev, pd_tau_pprev*q_velo_pprev

                    ff_power_val_curr, pd_power_val_curr   = torch.abs(torch.sum(ff_power_curr)), torch.abs(torch.sum(pd_power_curr))
                    ff_power_val_prev, pd_power_val_prev   = torch.abs(torch.sum(ff_power_prev)), torch.abs(torch.sum(pd_power_prev))
                    ff_power_val_pprev, pd_power_val_pprev = torch.abs(torch.sum(ff_power_pprev)), torch.abs(torch.sum(pd_power_pprev))

                    # Calculate a minimum effort gate
                    #    detach the gate values from the comp-graph, do not need gradients flowing through this
                    curr_effort_gate  = F.tanh(ff_power_val_curr.detach().clone() + pd_power_val_curr.detach().clone())
                    prev_effort_gate  = F.tanh(ff_power_val_prev.detach().clone() + pd_power_val_prev.detach().clone())
                    pprev_effort_gate = F.tanh(ff_power_val_pprev.detach().clone() + pd_power_val_pprev.detach().clone())

                    # Now create losses that seek to align power generation
                    #    using cosine similarity to enforce power expenditures that don't conflict 
                    task_alignment_curr  = torch.clamp(-F.cosine_similarity(ff_power_curr,  pd_power_curr),  min=0.0)
                    task_alignment_prev  = torch.clamp(-F.cosine_similarity(ff_power_prev,  pd_power_prev),  min=0.0)
                    task_alignment_pprev = torch.clamp(-F.cosine_similarity(ff_power_pprev, pd_power_pprev), min=0.0)

                    task_align_loss = torch.mean(torch.square(curr_effort_gate*task_alignment_curr + \
                                                              prev_effort_gate*task_alignment_prev + \
                                                              pprev_effort_gate*task_alignment_pprev))
                    
                    # # Finally, create a loss based on achiving a desired ratio of mechanical power.
                    # tagret_ratio = 1.8
                    # ratio_curr  =  ff_power_val_curr  / (pd_power_val_curr + 1e-6)
                    # ratio_prev  =  ff_power_val_prev  / (pd_power_val_prev + 1e-6)
                    # ratio_pprev =  ff_power_val_pprev / (pd_power_val_pprev + 1e-6)

                    # curr_error  =  torch.log((ratio_curr + 1e-6)  / (tagret_ratio + 1e-6))
                    # prev_error  =  torch.log((ratio_prev + 1e-6)  / (tagret_ratio + 1e-6))
                    # pprev_error =  torch.log((ratio_pprev + 1e-6) / (tagret_ratio + 1e-6))

                    # total_ratio_error = torch.square(curr_effort_gate*curr_error + prev_effort_gate*prev_error + pprev_effort_gate*pprev_error)
                    # normalized_ratio_error = total_ratio_error / (1+total_ratio_error)

                    # task_ratio_loss = torch.mean(normalized_ratio_error)

                
                ###
                #   END loss calculations, start gradient updates
                ### 
                if self.pinn_weight > 0.0 and use_pinn:
                    ppo_losses = [pos_loss+tau_loss,
                                  self.pinn_weight * pinn_loss,
                                  self.pinn_weight * task_align_loss]
                                #   self.pinn_weight * task_ratio_loss]
                else:
                    ppo_losses = [pos_loss+tau_loss]

                ###
                #   Perform encoder update step...
                ###
                self.decoder.eval()
                # self.enc_optimizer.zero_grad()
                # mean_latent, logvar_latent, cenet_latent, cenet_torso_velo = self.actor_critic.context_encoder(obs_hist_batch)

                dec_input = torch.cat((cenet_latent, cenet_torso_velo), dim=-1)
                enc_update_obs_decode = self.decoder(dec_input)
                
                grf_target.requires_grad = False
                obs_target.requires_grad = False
                
                # decode_target = torch.cat((obs_target, grf_target), dim=-1)
                decode_target = obs_target
                vel_target.requires_grad = False
                
                with torch.no_grad():
                    all_enc_obs_targets.extend(decode_target.detach().cpu().numpy())
                    all_enc_recons.extend(enc_update_obs_decode.clone().detach().cpu().numpy())

                # autoenc_loss = (nn.MSELoss()(cenet_torso_velo,vel_target) + nn.MSELoss()(enc_update_obs_decode,decode_target) + beta*(-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())))/self.num_mini_batches
                vel_pred_error = F.mse_loss(cenet_torso_velo,vel_target)
                recon_error    = F.mse_loss(enc_update_obs_decode,decode_target)
                kl_div         = (-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp()))
                autoenc_loss = vel_pred_error + recon_error + beta*kl_div
                
                ###
                #  Propigate gradients and update
                ### 
                
                # PCGrad - back-propigate the loss
                if use_pinn and self.pinn_weight > 0:
                    self.act_optimizer.pc_backward_pinn(ppo_losses)
                else:
                    self.act_optimizer.pc_backward(ppo_losses)
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.act_optimizer.step()
                # torch.cuda.empty_cache()
                
                autoenc_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.context_encoder.parameters(), self.max_grad_norm)
                self.enc_optimizer.step()
                # torch.cuda.empty_cache()

                ###
                #  Perfrom a separate update on the decoder....
                ###
                self.decoder.train()
                self.actor_critic.eval()
                self.decoder_optimizer.zero_grad()
                
                dec_out = self.decoder(dec_input.detach())
                dec_loss = F.mse_loss(dec_out, decode_target)
                dec_loss.backward()
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)
                self.decoder_optimizer.step()
                torch.cuda.empty_cache()

                # self.spectral_normalization(self.actor_critic)
                # self.spectral_normalization(self.decoder)


                mean_pos_value_loss += pos_value_loss.item()
                mean_pos_surrogate_loss += pos_surrogate_loss.item()
                mean_tau_value_loss += tau_value_loss.item()
                mean_tau_surrogate_loss += tau_surrogate_loss.item()
                mean_autoenc_loss += autoenc_loss.item()
                mean_vel_loss += vel_pred_error.item()
                mean_recon_loss += recon_error.item()
                mean_kld_loss += kl_div.item()
                mean_decoder_loss += dec_loss.item()
                if self.pinn_weight > 0.0:
                    mean_pinn_loss += pinn_loss.item() + task_align_loss.item() # + task_ratio_loss.item()
                else:
                    mean_pinn_loss += pinn_loss + task_align_loss # + task_ratio_loss
                    
        if use_pinn and itr > self.pinn_init:
            self.num_pinn_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_pos_value_loss /= num_updates
        mean_pos_surrogate_loss /= num_updates
        mean_tau_value_loss /= num_updates
        mean_tau_surrogate_loss /= num_updates
        mean_autoenc_loss /= num_updates
        mean_decoder_loss /= num_updates
        mean_kld_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_pinn_loss /= num_updates

        # Calculate the total bootstrapping probability over the performance of the autoencoder on all of the above
        mean_pred = np.mean(all_enc_obs_targets, axis=0)
        mean_pred_error = np.mean(np.square(mean_pred - all_enc_obs_targets))
        actual_pred_error = np.mean(np.square(np.array(all_enc_recons) - np.array(all_enc_obs_targets)))
        ratio = mean_pred_error / (actual_pred_error * self.boot_mult)
        pboot = np.tanh(ratio)

        # Use the (scaled) ratio of mean-prediction performance to actual prediction performance
        #     to determine if encoder bootstrapping is performed.
        self.use_boot = random.random() < pboot
        print(self.use_boot)
        
        with torch.no_grad():
            print(self.actor_critic.act_pos_2_tau_h1.scale.weight.norm())
            print(self.actor_critic.act_tau_2_pos_h1.scale.weight.norm())
            print(self.actor_critic.act_pos_2_tau_h2.scale.weight.norm())
            print(self.actor_critic.act_tau_2_pos_h2.scale.weight.norm())

        self.storage.clear()

        return mean_pos_value_loss, mean_pos_surrogate_loss, mean_tau_value_loss, \
               mean_tau_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, \
               mean_vel_loss, mean_recon_loss, mean_kld_loss, mean_pinn_loss
