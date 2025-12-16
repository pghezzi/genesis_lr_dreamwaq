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

import time
import os
from collections import deque
import statistics
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPODynamic
from rsl_rl.modules import ActorCritic_Dynamic, ContextDecoder
from rsl_rl.env import VecEnv


class OnPolicyRunnerDynamic:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        torch.autograd.set_detect_anomaly(True)
        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        
        cenet_input_dim = self.env.num_obs * self.env.num_obs_hist

        actor_critic: ActorCritic_Dynamic = actor_critic_class(self.env.num_obs,
                                                               num_critic_obs,
                                                               self.env.num_actions,
                                                               self.policy_cfg["actor_shared_dim"],
                                                               self.policy_cfg["actor_branch_layers"],
                                                               self.policy_cfg["critic_layers"],
                                                               cenet_input_dim,
                                                               self.policy_cfg["cenet_enc_latent_dim"],
                                                               self.policy_cfg["cenet_velo_dim"],
                                                               self.policy_cfg["cenet_enc_layers"],
                                                               self.policy_cfg["dropout"],
                                                               self.policy_cfg["activation"],
                                                               self.policy_cfg["init_noise_std"]).to(self.device)
        
        decoder = ContextDecoder(self.policy_cfg["cenet_dec_input_dim"],
                                 self.policy_cfg["cenet_dec_layers"],
                                 self.policy_cfg["cenet_dec_out_dim"],
                                 self.policy_cfg["dropout"]).to(self.device)
        
        print("Created Parallel Actor-Critic Model. Parameter Count: ", np.sum(p.numel() for p in actor_critic.parameters() if p.requires_grad))

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        self.alg: PPODynamic = alg_class(actor_critic, decoder, 
                                         pinn_lambda=self.policy_cfg["pinn_loss_weight"], 
                                         pinn_warmup=self.policy_cfg["pinn_warmup"], 
                                         pinn_init_steps=self.policy_cfg["pinn_init_steps"],
                                         device=self.device, **self.alg_cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_obs_hist*self.env.num_obs], \
                              [self.env.num_actions], [self.policy_cfg["cenet_velo_dim"]], [self.cfg["grf_dim"]], [self.env.wb_dim])

        if "pretrained_path" in self.policy_cfg.keys():
            self._load_pretrained_model()

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.create_async_pino_workers()

        _, _ = self.env.reset()

    # 'action': action_network.state_dict(),
    # 'decoder': decoder.state_dict(),
    def _load_pretrained_model(self):
        pretrained_path = self.policy_cfg["pretrained_path"]
        print(pretrained_path)
        loaded_dict = torch.load(pretrained_path)
        # Load the pretrained action-network and encoder
        self.alg.actor_critic.load_state_dict(loaded_dict['action'])
        # re-initalize the critic network weights which where not pretrained (to be safe)
        self.alg.actor_critic._init_critic_weights()
        # Load the pretrained decoder network
        self.alg.decoder.load_state_dict(loaded_dict['decoder'])
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        
        obs,obs_hist,torso_velo = self.env.get_observations()
        self.env.step_tradeoff_curriculum()
        if self.env.use_reward_curriculum:
            self.env.step_reward_curriculum()
        
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        
        obs, critic_obs, obs_hist, torso_velo = obs.to(self.device), critic_obs.to(self.device),obs_hist.to(self.device), torso_velo.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        pos_rewbuffer = deque(maxlen=100)
        tau_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_pos_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_tau_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    # extract previous observations (obs_{t-1}) BEFORE running env.step()
                    #     as env.step() takes the obs_t (used below to get a_t) and sets obs_{t-1} = obs_t before computing obs_{t+1}
                    #     NOTE - tracking the last obs across episode resets is handled in the env class
                    prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist = self.env.get_prev_obs()
                    prev_obs, prev_obs_hist = prev_obs.to(self.device), prev_obs_hist.to(self.device)
                    pprev_obs, pprev_obs_hist = pprev_obs.to(self.device), pprev_obs_hist.to(self.device)

                    # Call the algorithms act() method to store current transition data and predict actions
                    actions = self.alg.act(obs, critic_obs, obs_hist, torso_velo, prev_obs, prev_obs_hist, pprev_obs, pprev_obs_hist) # obs_t, (obs_t-1)
                         
                    # Submit the predicted action and extract the resulting state... 
                    obs, privileged_obs, obs_hist, pos_rewards, tau_rewards, dones, infos, grfs = self.env.step(actions)  # obs_t+1  (obs_t)
                    
                    # Create privileged obs
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    
                    # get the PINN specific data
                    gt_forces, mass_mats, bias_vecs, torso_acc = self.env.get_pinn_wb_dynamics()
                    gt_forces, mass_mats, bias_vecs, torso_acc = gt_forces.to(self.device), mass_mats.to(self.device), bias_vecs.to(self.device), torso_acc.to(self.device)

                    # move everything to the correct device
                    obs, critic_obs, obs_hist, pos_rewards, tau_rewards, dones, grfs = obs.to(self.device), critic_obs.to(self.device), \
                        obs_hist.to(self.device), pos_rewards.to(self.device), tau_rewards.to(self.device), dones.to(self.device), grfs.to(self.device)

                    # Log the labels associated with the context decoder as well as the typical stuff
                    self.alg.process_env_step(pos_rewards, tau_rewards, dones, infos, grfs, obs, gt_forces, mass_mats, bias_vecs, torso_acc)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_pos_reward_sum += pos_rewards
                        cur_tau_reward_sum += tau_rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        pos_rewbuffer.extend(cur_pos_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        tau_rewbuffer.extend(cur_tau_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_pos_reward_sum[new_ids] = 0
                        cur_tau_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_pos_value_loss, mean_pos_surrogate_loss, mean_tau_value_loss, mean_tau_surrogate_loss, mean_autoenc_loss, mean_decoder_loss, mean_pinn_loss = self.alg.update(self.env._get_pinn_actions,self.env.dt,self.env.num_iters)

            self.env.step_tradeoff_curriculum()
            if self.env.use_reward_curriculum:
                self.env.step_reward_curriculum()
            self.env.num_iters += 1

            # # enable additional domain randomizations
            # if (self.env.cfg.domain_rand.enable_additional_ratio < (float(self.env.num_iters)/float(tot_iter))):
            #     self.env.cfg.domain_rand.randomize_joint_armature = True
            #     self.env.cfg.domain_rand.randomize_joint_stiffness = True
            #     self.env.cfg.domain_rand.randomize_joint_damping = True
            
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

        # Learning is done, shutdown the async. pinocchio workers
        self.env.shutdown_asynic_pino_workers()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        # TODO - update this as well
        mean_pos_std = self.alg.actor_critic.std_pos.mean()
        mean_tau_std = self.alg.actor_critic.std_tau.mean()
        
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        self.writer.add_scalar('Loss/autoenc_function', locs['mean_autoenc_loss'], locs['it'])
        self.writer.add_scalar('Loss/decoder_function', locs['mean_decoder_loss'], locs['it'])
        self.writer.add_scalar('Loss/pos_value_function', locs['mean_pos_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/pos_surrogate', locs['mean_pos_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/pos_learning_rate', self.alg.pos_learning_rate, locs['it'])
        self.writer.add_scalar('Loss/tau_value_function', locs['mean_tau_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/tau_surrogate', locs['mean_tau_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/tau_learning_rate', self.alg.tau_learning_rate, locs['it'])
        self.writer.add_scalar('Loss/pinn_loss', locs['mean_pinn_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_pos_noise_std', mean_pos_std.item(), locs['it'])        
        self.writer.add_scalar('Policy/mean_tau_noise_std', mean_tau_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        
        if len(locs['pos_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_pos_reward', statistics.mean(locs['pos_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_tau_reward', statistics.mean(locs['tau_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_pos_reward/time', statistics.mean(locs['pos_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_tau_reward/time', statistics.mean(locs['tau_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['pos_rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'PINN loss:':>{pad}} {locs['mean_pinn_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Decoder function loss:':>{pad}} {locs['mean_decoder_loss']:.4f}\n"""
                          f"""{'Pos Value function loss:':>{pad}} {locs['mean_pos_value_loss']:.4f}\n"""
                          f"""{'Pos Surrogate loss:':>{pad}} {locs['mean_pos_surrogate_loss']:.4f}\n"""
                          f"""{'Tau Value function loss:':>{pad}} {locs['mean_tau_value_loss']:.4f}\n"""
                          f"""{'Tau Surrogate loss:':>{pad}} {locs['mean_tau_surrogate_loss']:.4f}\n"""
                          f"""{'Mean pos action noise std:':>{pad}} {mean_pos_std.item():.2f}\n"""
                          f"""{'Mean tau action noise std:':>{pad}} {mean_tau_std.item():.2f}\n"""
                          f"""{'Mean Pos reward:':>{pad}} {statistics.mean(locs['pos_rewbuffer']):.2f}\n"""
                          f"""{'Mean Tau reward:':>{pad}} {statistics.mean(locs['tau_rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'PINN loss:':>{pad}} {locs['mean_pinn_loss']:.4f}\n"""
                          f"""{'Autoenc function loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Decoder function loss:':>{pad}} {locs['mean_decoder_loss']:.4f}\n"""
                          f"""{'Pos Value function loss:':>{pad}} {locs['mean_pos_value_loss']:.4f}\n"""
                          f"""{'Pos Surrogate loss:':>{pad}} {locs['mean_pos_surrogate_loss']:.4f}\n"""
                          f"""{'Tau Value function loss:':>{pad}} {locs['mean_tau_value_loss']:.4f}\n"""
                          f"""{'Tau Surrogate loss:':>{pad}} {locs['mean_tau_surrogate_loss']:.4f}\n"""
                          f"""{'Mean pos action noise std:':>{pad}} {mean_pos_std.item():.2f}\n"""
                          f"""{'Mean tau action noise std:':>{pad}} {mean_tau_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'act_optimizer_state_dict': self.alg.act_optimizer.optimizer.state_dict(),
            'enc_optimizer_state_dict': self.alg.enc_optimizer.optimizer.state_dict(),
            # 'act_optimizer_state_dict': self.alg.act_optimizer.state_dict(),
            # 'enc_optimizer_state_dict': self.alg.enc_optimizer.state_dict(),
            'decoder_state_dict': self.alg.decoder.state_dict(),
            'decoder_opt_state_dict': self.alg.decoder_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        # Load actor/critic model(s)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Load optimizer(s)
        if load_optimizer:
            self.alg.act_optimizer.optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
            self.alg.enc_optimizer.optimizer.load_state_dict(loaded_dict['enc_optimizer_state_dict'])
            # self.alg.act_optimizer.load_state_dict(loaded_dict['act_optimizer_state_dict'])
            # self.alg.enc_optimizer.load_state_dict(loaded_dict['enc_optimizer_state_dict'])
            self.alg.decoder_optimizer.load_state_dict(loaded_dict['decoder_opt_state_dict'])
        # Load the VAE decoder model...
        self.alg.decoder.load_state_dict(loaded_dict['decoder_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
