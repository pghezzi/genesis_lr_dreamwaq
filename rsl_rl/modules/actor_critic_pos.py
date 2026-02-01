from __future__ import annotations

from typing import Tuple, List, Dict, Any
from torch.distributions import Normal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



###
#
# adaLN Layers
#
###
class _ShiftScaleMod(nn.Module):
    """Adaptive shift-scale modulation layer with learnable parameters.

    Implements: output = x * scale(c) + shift(c)
    where c is a conditioning vector and scale/shift are learned transformations.

    Args:
        dim (int): Feature dimension for both input and conditioning.
    """

    def __init__(self, dim: int, activation) -> None:
        super().__init__()
        self.act = get_activation(activation)
        self.scale = nn.Linear(dim, 1)
        self.shift = nn.Linear(dim, 1)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Applies adaptive scaling and shifting.

        Args:
            x: Input tensor of shape [batch_size, dim]
            c: Conditioning tensor of shape [batch_size, dim]
        Returns:
            Modulated tensor of same shape as input
        """
        c = self.act(c)
        return x * self.scale(c) + self.shift(c)

    def reset_parameters(self) -> None:
        """Initializes weights with Xavier uniform and zeros biases."""
        # nn.init.zeros_(self.scale.weight)
        # nn.init.zeros_(self.shift.weight)
        nn.init.uniform_(self.scale.weight, -1e-10, 1e-10)
        nn.init.uniform_(self.shift.weight, -1e-10, 1e-10)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

###
#
#   Context Encoder/Decoder models used to provide conditioning input
#
###
class ContextEncoder(nn.Module):
    """VAE-style encoder for processing context information with velocity prediction.

    Encodes high-dimensional context into a latent distribution while simultaneously
    predicting torso velocity. Uses ELU activations and Xavier initialization.

    Args:
        context_input_dim (int): Dimension of input context features. Default: 230.
        context_layer_sizes (List[int]): Sizes of hidden layers. Default: [128, 64, 32].
        context_latent_size (int): Dimension of latent space. Default: 16.
        context_torso_velo_size (int): Dimension of velocity output. Default: 3.
        device (str): Device for tensor operations. Default: "cpu".
    """
    def __init__(
        self,
        context_input_dim: int = 230,
        context_layer_sizes: List[int] = [128, 64],
        context_latent_size: int = 16,
        context_torso_velo_size: int = 3,
        dropout: float = 0.1,
        activation: str = 'swish',
        device: str = "cpu"
    ) -> None:
        super().__init__()

        # VAE-style context encoder layers
        # Input Layer
        self.ce_in = nn.Linear(context_input_dim, context_layer_sizes[0])
        # Hidden Layers
        self.ce_h1 = nn.Linear(context_layer_sizes[0], context_layer_sizes[1])
        # Output Layers
        self.ce_out_mean = nn.Linear(context_layer_sizes[1], context_latent_size)
        self.ce_out_var = nn.Linear(context_layer_sizes[1], context_latent_size)

        self.ce_velo_mean = nn.Linear(context_layer_sizes[1], context_torso_velo_size)
        self.ce_velo_var  = nn.Linear(context_layer_sizes[1], context_torso_velo_size)

        self.qr_h1 = nn.Linear(context_layer_sizes[1], context_layer_sizes[1])
        self.qr_h2 = nn.Linear(context_layer_sizes[1], context_layer_sizes[1])
        
        
        self.ce_qr_mean = nn.Linear(context_layer_sizes[1], 12)
        self.ce_qr_var = nn.Linear(context_layer_sizes[1], 12)

        self.activation = get_activation(activation)

        # self.ce_timestep = nn.Linear(context_layer_sizes[2], 1)

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

        self.device = device
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.ce_in, self.ce_h1,
                     self.ce_out_mean, self.ce_out_var, 
                     self.ce_velo_mean, self.ce_velo_var,
                     self.ce_qr_mean, self.ce_qr_var,
                     self.qr_h1, self.qr_h2]:
            
            nn.init.xavier_uniform_(layer.weight)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through encoder
        x = self.activation(self.ce_in(X_C))
        # x = self.drop_1(x)
        x = self.activation(self.ce_h1(x))
        # x = self.drop_2(x)

        x_qr = self.activation(self.qr_h1(x))
        x_qr = self.activation(self.qr_h2(x_qr))

        return self.ce_out_mean(x), self.ce_out_var(x), self.ce_velo_mean(x), self.ce_velo_var(x), self.ce_qr_mean(x_qr), self.ce_qr_var(x_qr)

    def reparameterization_trick(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from latent space using reparameterization trick.

        Args:
            mean: Latent space mean
            logvar: Latent space log variance

        Returns:
            Sampled latent vector
        """
        epsilon = torch.randn_like(logvar).to(logvar.device)
        return mean + torch.exp(0.5 * logvar) * epsilon

    def forward(self, X_C: torch.Tensor):
        """Complete forward pass including encoding and sampling.

        Args:
            X_C: Input context tensor

        Returns:
            Tuple containing:
                - mean: Latent space mean
                - logvar: Latent space log variance
                - z: Sampled latent vector
                - torso_velo: Predicted velocity
        """
        mean, logvar, v_mean, v_logvar, q_ref_mean, q_ref_var = self.encode(X_C)
        z = self.reparameterization_trick(mean, logvar)
        torso_velo = self.reparameterization_trick(v_mean, v_logvar)
        q_ref = self.reparameterization_trick(q_ref_mean,q_ref_var)
        return mean, logvar, z, torso_velo, q_ref

class ContextDecoder(nn.Module):
    """Decoder network for reconstructing next state from latent representation and velocity.
    
    Takes a latent vector and torso velocity as input, processes through two ELU-activated
    hidden layers, and outputs a predicted next state. Uses Xavier uniform initialization.
    """

    def __init__(
            self,
            input_dim: int = 19,
            layers: List[int] = [64,128],
            decode_dim: int = 57,
            dropout: float = 0.1) -> None:
        super().__init__()


        # Network architecture
        self.dec_in = nn.Linear(input_dim, layers[0])
        self.dec_h1 = nn.Linear(layers[0], layers[1])
        self.dec_out = nn.Linear(layers[1], decode_dim)

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.dec_in, self.dec_h1, self.dec_out]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder network.

        Args:
            condition: Latent vector of shape (batch_size, latent_dim+velo_dim)
        Returns:
            Reconstructed next state of shape (batch_size, decode_dim)
        """
        # Process through network with ELU activations
        x = F.elu(self.dec_in(condition))
        x = self.drop_1(x)
        x = F.elu(self.dec_h1(x))
        x = self.drop_2(x)
        return self.dec_out(x)


# class ActorCritic_Pos(nn.Module):
#     def __init__(self, 
#                  num_actor_obs=69, 
#                  num_critic_obs=95, 
#                  num_actions=12,
#                  actor_shared_dim=512,
#                  actor_branch_layers=[256,128,64],
#                  critic_layers=[512,256,128,64],
#                  cenet_in_dim=350,
#                  cenet_latent_dim=29,
#                  cenet_velo_dim=3, 
#                  cenet_enc_layers=[256,128,64],
#                  dropout=0.1,
#                  activation="elu", 
#                  init_noise_std=1.0,):
#         super().__init__()

#         # Construct the context encoder network
#         self.context_encoder = ContextEncoder(context_input_dim=cenet_in_dim,
#                                               context_layer_sizes=cenet_enc_layers,
#                                               context_latent_size=cenet_latent_dim,
#                                               context_torso_velo_size=cenet_velo_dim,
#                                               dropout=dropout,
#                                               activation=activation)
        
#         # Get the activation function used by the actor and critic networks
#         self.activation = get_activation(activation)

#         self.init_noise_std = init_noise_std

#         ###
#         #  Construct the layers for the actor network
#         ###
#         # Shared layer between output branches
#         actor_input_dim = num_actor_obs + cenet_latent_dim + cenet_velo_dim
#         self.actor_shared_input = nn.Linear(actor_input_dim, actor_shared_dim)

#         # Now create separate branches for position and torque control
#         #     Branch for Position Control
#         self.act_pos_h1  = nn.Linear(actor_shared_dim, actor_branch_layers[0])
#         self.act_pos_h2  = nn.Linear(actor_branch_layers[0], actor_branch_layers[1])
#         self.act_pos_out = nn.Linear(actor_branch_layers[1], num_actions)

#         ###
#         #  Construct layers for the critic network
#         ###
#         self.pos_critic_in  = nn.Linear(num_critic_obs, actor_shared_dim)
#         self.pos_critic_h1  = nn.Linear(actor_shared_dim, critic_layers[0])
#         self.pos_critic_h2  = nn.Linear(critic_layers[0], critic_layers[1])
#         self.pos_critic_out = nn.Linear(critic_layers[1], 1)

#         #     Branch for torque control
#         self.act_tau_h1 = nn.Linear(actor_shared_dim, actor_branch_layers[0])
#         self.act_tau_h2  = nn.Linear(actor_branch_layers[0], actor_branch_layers[1])
#         self.act_tau_out = nn.Linear(actor_branch_layers[1], num_actions)

#         # Now create FiLM layers for sharing info. between branches
#         #     Applied after h1
#         self.act_pos_2_tau_h1 = _ShiftScaleMod(dim=actor_branch_layers[0]*2, activation=activation)
#         self.act_tau_2_pos_h1 = _ShiftScaleMod(dim=actor_branch_layers[0]*2, activation=activation)
#         #     Applied after h2
#         self.act_pos_2_tau_h2 = _ShiftScaleMod(dim=actor_branch_layers[1]*2, activation=activation)
#         self.act_tau_2_pos_h2 = _ShiftScaleMod(dim=actor_branch_layers[1]*2, activation=activation)

#         self.act_pos_layernorm_1 = nn.LayerNorm(actor_branch_layers[0])
#         self.act_tau_layernorm_1 = nn.LayerNorm(actor_branch_layers[0])

#         self.act_pos_layernorm_2 = nn.LayerNorm(actor_branch_layers[1])
#         self.act_tau_layernorm_2 = nn.LayerNorm(actor_branch_layers[1])

#         # Used to track these values during training....
#         #     These values will not be used during inference (sim or real)
#         self.cenet_mean = None 
#         self.cenet_logvar = None
#         self.cenet_z = None
#         self.cenet_torso_velo = None

#         self.mean_pos = None
#         self.mean_tau = None

#         self.std_tau = nn.Parameter(init_noise_std * torch.ones(num_actions))
#         self.std_pos = nn.Parameter(init_noise_std * torch.ones(num_actions))
        
#         self.num_actions = num_actions
#         self.distribution_pos = None
        
#         # disable args validation for speedup
#         Normal.set_default_validate_args = False

#         # initalize the weights
#         # self._init_weights()

#     def _init_weights(self):
#         # Xavier for linears, zeros for biases
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 # nn.init.xavier_uniform_(m.weight)
#                 nn.init.orthogonal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)

#         # # Optionally set small initial output weights (to reduce initial action magnitude)
#         nn.init.uniform_(self.act_pos_out.weight, -3e-2, 3e-2)
#         nn.init.zeros_(self.act_pos_out.bias)
        

#     def get_optim_groups(self, weight_decay: float = 1e-4, strong_decay: float = 1e-1):
#         """Separate parameters into groups:
#             (1) does not use weight decay
#             (2) uses extra strong weight decay (FiLM layers)
#             (3) shared parameters with weight decay
#             (4) parameters for the position control output with weight decay
#             (5) parameters for the torque control output with weight decay
        
#         Args:
#             weight_decay (float): Weight decay value for regularization. Default: 1e-4.
            
#         Returns:
#             List of parameter groups for optimizer initialization.
#         """
#         critic_set = set()
#         tau_decay = set()
#         pos_decay = set()
#         shared_decay = set()
#         no_decay  = set()
#         special_decay = set()
#         encoder = set()
#         whitelist = (nn.Linear, nn.MultiheadAttention)
#         blacklist = (nn.LayerNorm, nn.Embedding, nn.Sequential, nn.Parameter)

#         for mn, m in self.named_modules():
#             for pn, p in m.named_parameters():
#                 fpn = f"{mn}.{pn}" if mn else pn  # full param name
#                 if pn.endswith("bias") or pn.startswith("bias"):
#                     no_decay.add(fpn)
#                 elif pn.endswith("weight"):
#                     if isinstance(m, blacklist):
#                         no_decay.add(fpn)
                    
#                     elif isinstance(m, whitelist):
#                         if "_2_" in fpn:
#                             # Here is the name contains a "2" then it is a cross-conditioning
#                             #    FILM layer, and we want a stronger weight reg on these values
#                             special_decay.add(fpn)
#                         elif "shared" in fpn:
#                             # We do not want the shared layer's learning rate to be modified during multi-task PPO updates
#                             shared_decay.add(fpn)
#                         elif "context_encoder" in fpn:
#                             encoder.add(fpn)
#                         elif "critic" in fpn:
#                             critic_set.add(fpn)
#                         else:
#                             if "pos" in fpn:
#                                 pos_decay.add(fpn)
#                             elif "tau" in fpn:
#                                 tau_decay.add(fpn)
#                             else:
#                                 # A catch for anything I missed above... 
#                                 shared_decay.add(fpn)

#         # for i in range(self.options["action_net"]["num_layers"]-1):
#         #     no_decay.update([f"noise_decoder.cross_field_scales_pos.{i}", f"noise_decoder.cross_field_scales_tau.{i}"])
#         no_decay.update([f"std_pos", f"std_tau"])
#         # shared_decay.update([f"std"])

#         # Validate parameter separation
#         param_dict   = {pn: p for pn, p in self.named_parameters()}
#         inter_params = pos_decay & tau_decay & shared_decay & no_decay & special_decay & encoder & critic_set
#         if inter_params:
#             raise ValueError(f"Parameters in all sets: {inter_params}")
#         missing_params = param_dict.keys() - (pos_decay | tau_decay | shared_decay | no_decay | special_decay | encoder | critic_set)
#         if missing_params:
#             raise ValueError(f"Parameters not categorized: {missing_params}")
        
#         # print(f"Parameters with extra strong weight decay{special_decay}")

#         params_act = [{"params": [param_dict[pn] for pn in sorted(pos_decay)],     "weight_decay": 0.0, "name":"pos_branch"},
#                       {"params": [param_dict[pn] for pn in sorted(tau_decay)],     "weight_decay": 0.0, "name":"tau_branch"},
#                       {"params": [param_dict[pn] for pn in sorted(shared_decay)],  "weight_decay": 0.0, "name":"shared"},
#                       {"params": [param_dict[pn] for pn in sorted(critic_set)],    "weight_decay": 0.0, "name":"critic"},
#                       {"params": [param_dict[pn] for pn in sorted(no_decay)],      "weight_decay": 0.0},
#                       {"params": [param_dict[pn] for pn in sorted(special_decay)], "weight_decay": strong_decay}]
        
#         params_enc = [{"params": [param_dict[pn] for pn in sorted(encoder)], "weight_decay": weight_decay, "name":"encoder"}]

#         return params_act, params_enc

#     def configure_optimizers(self,
#                              learning_rate: float = 1e-4,
#                              weight_decay: float = 1e-6,
#                              strong_decay: float = 1e-1,
#                              betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
#         """Configure the AdamW optimizer with parameter groups.

#         Standard weights in Linear/Attention layers - weight_decay
#         all bias terms - no decay
#         FiLM layer parameters - strong_decay
            
#         Returns:
#             Configured AdamW optimizer.
#         """
#         opt_groups_act, opt_groups_enc = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)
#         act_opt = torch.optim.AdamW(opt_groups_act, lr=learning_rate)
#         enc_opt = torch.optim.AdamW(opt_groups_enc, lr=2.0e-4)
#         return act_opt, enc_opt

#     def reset(self, dones=None):
#         pass

#     def forward(self):
#         raise NotImplementedError
    
#     # forward methods for the histroical context VAE
#     def cenet_enc_forward(self, obs_history):
#         mean, logvar, z, torso_velo = self.context_encoder(obs_history)
#         return mean, logvar, z, torso_velo

#     # Method for the forward method of the actor network, used mostly as an internal method
#     def actor_forward(self, current_obs):
#         # run the concatonated vector through the shared encoder layer
#         x = self.actor_shared_input(current_obs)
#         x = self.activation(x)

#         # Now run the two parallel branches
#         #     position
#         pos_latent = self.act_pos_h1(x)
#         #     torque
#         tau_latent = self.act_tau_h1(x)
#         #     now perform the cross-conditioning
#         #     FiLM layer has a built-in axtivation function
#         condition = torch.cat([pos_latent, tau_latent], dim=-1)
#         pos_latent = pos_latent + self.act_tau_2_pos_h1(tau_latent, condition)  # perform FiLM on pos_latent using tau_latent
#         tau_latent = tau_latent + self.act_pos_2_tau_h1(pos_latent, condition)  # perform FiLM on tau_latent using pos_latent
#         # Perform activation
#         pos_latent = self.activation(pos_latent)
#         tau_latent = self.activation(tau_latent)
#         # Perform layer normalization
#         pos_latent = self.act_pos_layernorm_1(pos_latent)
#         tau_latent = self.act_tau_layernorm_1(tau_latent)
        
#         # REPEAT
#         #     position
#         pos_latent = self.act_pos_h2(pos_latent)
#         #     torque
#         tau_latent = self.act_tau_h2(tau_latent)
#         #     now perform the cross-conditioning
#         condition = torch.cat([pos_latent, tau_latent], dim=-1)
#         pos_latent = pos_latent + self.act_tau_2_pos_h2(tau_latent, condition)  # perform FiLM on pos_latent using tau_latent
#         tau_latent = tau_latent + self.act_pos_2_tau_h2(pos_latent, condition)  # perform FiLM on tau_latent using pos_latent
#         # perform activation
#         pos_latent = self.activation(pos_latent)
#         tau_latent = self.activation(tau_latent)
#         # Perform layer normalization
#         pos_latent = self.act_pos_layernorm_2(pos_latent)
#         tau_latent = self.act_tau_layernorm_2(tau_latent)


#         # Now run the final output layers to get both action modalities
#         # act_pos_act = F.tanh(self.act_pos_out(pos_latent))
#         # act_tau_act = F.tanh(self.act_tau_out(tau_latent))
#         act_pos_act = self.act_pos_out(pos_latent)
#         act_tau_act = self.act_tau_out(tau_latent)

#         # print(act_pos_act)
#         # print(act_tau_act)

#         # CLAMP LOGITS
#         #    These SHOULD be generous clamp values...
#         # act_pos_act = torch.clamp(act_pos_act, -5.0, 5.0)
#         # act_tau_act = torch.clamp(act_tau_act, -5.0, 5.0)

#         if torch.isnan(act_pos_act).any():
#             with torch.no_grad():
#                 act_pos_act = torch.nan_to_num(act_pos_act, nan=0.0, posinf=0.0, neginf=0.0)

#         if torch.isnan(act_tau_act).any():
#             with torch.no_grad():
#                 act_tau_act = torch.nan_to_num(act_tau_act, nan=0.0, posinf=0.0, neginf=0.0)

#         # out = torch.cat([act_pos_act, act_tau_act], dim=-1)
#         # out = torch.nan_to_num(out, nan=0.0)

#         # print(act_pos_act)
#         # print(act_tau_act)
#         # print("----------------------------------\n")

#         return act_pos_act, act_tau_act

#     # Functions that are specific to PPO training
#     @property
#     def pos_action_mean(self):
#         return self.distribution_pos.mean

#     @property
#     def pos_action_std(self):
#         return self.distribution_pos.stddev

#     @property
#     def pos_entropy(self):
#         return self.distribution_pos.entropy().sum(dim=-1)
    
#     def get_pos_actions_log_prob(self, actions):
#         return self.distribution_pos.log_prob(actions).sum(dim=-1)

#     def update_distribution(self, curr_obs):
#         mean_pos, tau = self.actor_forward(curr_obs)
#         self.mean_pos = mean_pos
#         self.mean_tau = tau

#         # # Clamp Global STD parameters to be safe
#         # std_pos = torch.clamp(self.std_pos, 0.05, 1.1)
        
#         self.std_pos.data.clamp_(0.05, 5.0)

#         # self.std.data.clamp_(0.2, 1.1)

#         self.distribution_pos = Normal(mean_pos, mean_pos * 0.0 + self.std_pos)
        
#         return tau

#     # method used during simulated training
#     def act(self, obs, obs_history, **kwargs):
#         # Call the forward method of the context encoder
#         mean, logvar, z, torso_velo = self.cenet_enc_forward(obs_history)
        
#         # create the actors observation
#         current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
#         # Upated the PPO training distribution
#         self.update_distribution(current_obs)
        
#         # log context-encoder values to be used in PPO class for calculating context encoder network
#         self.cenet_mean = mean
#         self.cenet_logvar = logvar
#         self.cenet_z = z
#         self.cenet_torso_velo = torso_velo

#         pos_sample = self.distribution_pos.sample()
        
#         # return a sample from the distribution to be executed in simulation
#         return pos_sample
    

#     # method used during simulated training
#     def act_bootmask(self, obs, obs_history, **kwargs):
#         # Call the forward method of the context encoder
#         mean, logvar, z, torso_velo = self.cenet_enc_forward(obs_history)
        
#         # Mask the latent/velo from the encoder with zeros
#         boot_mask = torch.zeros((z.shape[0], (z.shape[1] + torso_velo.shape[1])), device=obs.device)

#         # create the actors observation
#         current_obs = torch.cat((obs,boot_mask), dim=-1)   
        
#         # Upated the PPO training distribution
#         self.update_distribution(current_obs)
        
#         # log context-encoder values to be used in PPO class for calculating context encoder network
#         self.cenet_mean = mean
#         self.cenet_logvar = logvar
#         self.cenet_z = z
#         self.cenet_torso_velo = torso_velo

#         pos_sample = self.distribution_pos.sample()
        
#         # return a sample from the distribution to be executed in simulation
#         return pos_sample
    
#     # Method using during simulated inference
#     def act_inference(self,obs,obs_history):
#         # Call the forward method of the context encoder
#         _, _, z, torso_velo = self.cenet_enc_forward(obs_history)
        
#         # create the actors observation
#         current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
#         # call the actors forward method and return it's results
#         actions_pos, _ = self.actor_forward(current_obs)

#         return actions_pos

#     # Method to run inference on hardware WITHOUT logging the VAE's outputs
#     @torch.jit.export
#     def act_inference_deploy(self, obs, obs_history):
#         # Call the forward method of the context encoder
#         _, _, z, torso_velo = self.cenet_enc_forward(obs_history)
        
#         # create the actors observation
#         current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
#         # call the actors forward method and return it's results
#         actions_pos = self.actor_forward(current_obs)
#         return actions_pos
    
#     @torch.jit.export
#     def act_inference_deploy_log(self, obs, obs_history):
#         # Call the forward method of the context encoder
#         _, _, z, torso_velo = self.cenet_enc_forward(obs_history)
        
#         # create the actors observation
#         current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
#         # call the actors forward method and return it's results
#         actions_pos = self.actor_forward(current_obs)

#         return actions_pos, z, torso_velo

#     # Forward method for calculating the value of the current state
#     #     using the privilged critic observation
#     def evaluate_pos(self, critic_observations, **kwargs):
#         val = self.pos_critic_in(critic_observations)
#         val = self.activation(val)
#         val = self.pos_critic_h1(val)
#         val = self.activation(val)
#         val = self.pos_critic_h2(val)
#         val = self.activation(val)
#         # val = self.pos_critic_h3(val)
#         # val = self.activation(val)

#         return self.pos_critic_out(val)

class ActorCritic_Pos(nn.Module):
    def __init__(self, 
                 num_actor_obs=69, 
                 num_critic_obs=95, 
                 num_actions=12,
                 actor_shared_dim=512,
                 actor_branch_layers=[256,128,64],
                 critic_layers=[512,256,128,64],
                 cenet_in_dim=350,
                 cenet_latent_dim=29,
                 cenet_velo_dim=3, 
                 cenet_enc_layers=[256,128,64],
                 dropout=0.1,
                 activation="elu", 
                 init_noise_std=1.0,):
        super().__init__()

        # Construct the context encoder network
        self.context_encoder = ContextEncoder(context_input_dim=cenet_in_dim,
                                              context_layer_sizes=cenet_enc_layers,
                                              context_latent_size=cenet_latent_dim,
                                              context_torso_velo_size=cenet_velo_dim,
                                              dropout=dropout,
                                              activation=activation)
        
        # Get the activation function used by the actor and critic networks
        self.activation = get_activation(activation)

        self.init_noise_std = init_noise_std

        ###
        #  Construct the layers for the actor network
        ###
        # Shared layer between output branches
        actor_input_dim = num_actor_obs + cenet_latent_dim + cenet_velo_dim
        self.actor_shared_input = nn.Linear(actor_input_dim, actor_shared_dim)

        # Now create separate branches for position and torque control
        #     Branch for Position Control
        self.act_pos_h1  = nn.Linear(actor_shared_dim, actor_branch_layers[0])
        self.act_pos_h2  = nn.Linear(actor_branch_layers[0], actor_branch_layers[1])
        self.act_pos_out = nn.Linear(actor_branch_layers[1], num_actions)

        ###
        #  Construct layers for the critic network
        ###
        self.pos_critic_in  = nn.Linear(num_critic_obs, actor_shared_dim)
        self.pos_critic_h1  = nn.Linear(actor_shared_dim, critic_layers[0])
        self.pos_critic_h2  = nn.Linear(critic_layers[0], critic_layers[1])
        self.pos_critic_out = nn.Linear(critic_layers[1], 1)

        # Used to track these values during training....
        #     These values will not be used during inference (sim or real)
        self.cenet_mean = None 
        self.cenet_logvar = None
        self.cenet_z = None
        self.cenet_torso_velo = None
        self.cenet_q_ref = None

        self.mean_pos = None

        self.std_pos = nn.Parameter(init_noise_std * torch.ones(num_actions))
        
        self.num_actions = num_actions
        self.distribution_pos = None
        
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # initalize the weights
        # self._init_weights()

    def _init_weights(self):
        # Xavier for linears, zeros for biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # # Optionally set small initial output weights (to reduce initial action magnitude)
        nn.init.uniform_(self.act_pos_out.weight, -3e-2, 3e-2)
        nn.init.zeros_(self.act_pos_out.bias)
        

    def get_optim_groups(self, weight_decay: float = 1e-4, strong_decay: float = 1e-1):
        """Separate parameters into groups:
            (1) does not use weight decay
            (2) uses extra strong weight decay (FiLM layers)
            (3) shared parameters with weight decay
            (4) parameters for the position control output with weight decay
            (5) parameters for the torque control output with weight decay
        
        Args:
            weight_decay (float): Weight decay value for regularization. Default: 1e-4.
            
        Returns:
            List of parameter groups for optimizer initialization.
        """
        critic_set = set()
        tau_decay = set()
        pos_decay = set()
        shared_decay = set()
        no_decay  = set()
        special_decay = set()
        encoder = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding, nn.Sequential, nn.Parameter)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight"):
                    if isinstance(m, blacklist):
                        no_decay.add(fpn)
                    
                    elif isinstance(m, whitelist):
                        if "_2_" in fpn:
                            # Here is the name contains a "2" then it is a cross-conditioning
                            #    FILM layer, and we want a stronger weight reg on these values
                            special_decay.add(fpn)
                        elif "shared" in fpn:
                            # We do not want the shared layer's learning rate to be modified during multi-task PPO updates
                            shared_decay.add(fpn)
                        elif "context_encoder" in fpn:
                            encoder.add(fpn)
                        elif "critic" in fpn:
                            critic_set.add(fpn)
                        else:
                            if "pos" in fpn:
                                pos_decay.add(fpn)
                            elif "tau" in fpn:
                                tau_decay.add(fpn)
                            else:
                                # A catch for anything I missed above... 
                                shared_decay.add(fpn)

        # for i in range(self.options["action_net"]["num_layers"]-1):
        #     no_decay.update([f"noise_decoder.cross_field_scales_pos.{i}", f"noise_decoder.cross_field_scales_tau.{i}"])
        no_decay.update([f"std_pos"])
        # shared_decay.update([f"std"])

        # Validate parameter separation
        param_dict   = {pn: p for pn, p in self.named_parameters()}
        inter_params = pos_decay & tau_decay & shared_decay & no_decay & special_decay & encoder & critic_set
        if inter_params:
            raise ValueError(f"Parameters in all sets: {inter_params}")
        missing_params = param_dict.keys() - (pos_decay | shared_decay | no_decay | special_decay | encoder | critic_set)
        if missing_params:
            raise ValueError(f"Parameters not categorized: {missing_params}")
        
        # print(f"Parameters with extra strong weight decay{special_decay}")

        params_act = [{"params": [param_dict[pn] for pn in sorted(pos_decay)],     "weight_decay": 0.0, "name":"pos_branch"},
                      {"params": [param_dict[pn] for pn in sorted(shared_decay)],  "weight_decay": 0.0, "name":"shared"},
                      {"params": [param_dict[pn] for pn in sorted(critic_set)],    "weight_decay": 0.0, "name":"critic"},
                      {"params": [param_dict[pn] for pn in sorted(no_decay)],      "weight_decay": 0.0},
                      {"params": [param_dict[pn] for pn in sorted(special_decay)], "weight_decay": strong_decay}]
        
        params_enc = [{"params": [param_dict[pn] for pn in sorted(encoder)], "weight_decay": weight_decay, "name":"encoder"}]

        return params_act, params_enc

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-6,
                             strong_decay: float = 1e-1,
                             betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer with parameter groups.

        Standard weights in Linear/Attention layers - weight_decay
        all bias terms - no decay
        FiLM layer parameters - strong_decay
            
        Returns:
            Configured AdamW optimizer.
        """
        opt_groups_act, opt_groups_enc = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)
        act_opt = torch.optim.AdamW(opt_groups_act, lr=learning_rate)
        enc_opt = torch.optim.AdamW(opt_groups_enc, lr=2.0e-4)
        return act_opt, enc_opt

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    # forward methods for the histroical context VAE
    def cenet_enc_forward(self, obs_history):
        mean, logvar, z, torso_velo, q_ref = self.context_encoder(obs_history)
        return mean, logvar, z, torso_velo, q_ref

    # Method for the forward method of the actor network, used mostly as an internal method
    def actor_forward(self, current_obs):
        # run the concatonated vector through the shared encoder layer
        x = self.actor_shared_input(current_obs)
        x = self.activation(x)

        # Now run the two parallel branches
        #     position
        pos_latent = self.act_pos_h1(x)
        # Perform activation
        pos_latent = self.activation(pos_latent)
        
        # REPEAT
        #     position
        pos_latent = self.act_pos_h2(pos_latent)
        # perform activation
        pos_latent = self.activation(pos_latent)


        # Now run the final output layers to get both action modalities
        act_pos_act = self.act_pos_out(pos_latent)


        if torch.isnan(act_pos_act).any():
            with torch.no_grad():
                act_pos_act = torch.nan_to_num(act_pos_act, nan=0.0, posinf=0.0, neginf=0.0)

        return act_pos_act

    # Functions that are specific to PPO training
    @property
    def pos_action_mean(self):
        return self.distribution_pos.mean

    @property
    def pos_action_std(self):
        return self.distribution_pos.stddev

    @property
    def pos_entropy(self):
        return self.distribution_pos.entropy().sum(dim=-1)
    
    def get_pos_actions_log_prob(self, actions):
        return self.distribution_pos.log_prob(actions).sum(dim=-1)

    def update_distribution(self, curr_obs):
        mean_pos = self.actor_forward(curr_obs)
        self.mean_pos = mean_pos

        # # Clamp Global STD parameters to be safe
        # std_pos = torch.clamp(self.std_pos, 0.05, 1.1)
        
        self.std_pos.data.clamp_(0.05, 5.0)

        # self.std.data.clamp_(0.2, 1.1)

        self.distribution_pos = Normal(mean_pos, mean_pos * 0.0 + self.std_pos)
        
    # method used during simulated training
    def act(self, obs, obs_history, **kwargs):
        # Call the forward method of the context encoder
        mean, logvar, z, torso_velo, q_ref = self.cenet_enc_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)
        
        # log context-encoder values to be used in PPO class for calculating context encoder network
        self.cenet_mean = mean
        self.cenet_logvar = logvar
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo
        self.cenet_q_ref = q_ref

        pos_sample = self.distribution_pos.sample()
        
        # return a sample from the distribution to be executed in simulation
        return pos_sample
    

    # method used during simulated training
    def act_bootmask(self, obs, obs_history, **kwargs):
        # Call the forward method of the context encoder
        mean, logvar, z, torso_velo, q_ref = self.cenet_enc_forward(obs_history)
        
        # Mask the latent/velo from the encoder with zeros
        boot_mask = torch.zeros((z.shape[0], (z.shape[1] + torso_velo.shape[1])), device=obs.device)

        # create the actors observation
        current_obs = torch.cat((obs,boot_mask), dim=-1)   
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)
        
        # log context-encoder values to be used in PPO class for calculating context encoder network
        self.cenet_mean = mean
        self.cenet_logvar = logvar
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo
        self.cenet_q_ref = q_ref

        pos_sample = self.distribution_pos.sample()
        
        # return a sample from the distribution to be executed in simulation
        return pos_sample
    
    # Method using during simulated inference
    def act_inference(self,obs,obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo, q_ref = self.cenet_enc_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_pos = self.actor_forward(current_obs)

        return actions_pos, q_ref

    # Method to run inference on hardware WITHOUT logging the VAE's outputs
    @torch.jit.export
    def act_inference_deploy(self, obs, obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo, q_ref = self.cenet_enc_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_pos = self.actor_forward(current_obs)
        return actions_pos, q_ref
    
    @torch.jit.export
    def act_inference_deploy_log(self, obs, obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo = self.cenet_enc_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_pos = self.actor_forward(current_obs)

        return actions_pos, z, torso_velo

    # Forward method for calculating the value of the current state
    #     using the privilged critic observation
    def evaluate_pos(self, critic_observations, **kwargs):
        val = self.pos_critic_in(critic_observations)
        val = self.activation(val)
        val = self.pos_critic_h1(val)
        val = self.activation(val)
        val = self.pos_critic_h2(val)
        val = self.activation(val)
        # val = self.pos_critic_h3(val)
        # val = self.activation(val)

        return self.pos_critic_out(val)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "swish":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None
