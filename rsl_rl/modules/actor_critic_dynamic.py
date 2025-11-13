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

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)
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
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
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
        context_layer_sizes: List[int] = [128, 64, 32],
        context_latent_size: int = 16,
        context_torso_velo_size: int = 3,
        dropout: float = 0.1,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        if len(context_layer_sizes) != 3:
            raise ValueError("context_layer_sizes must contain exactly 3 values")

        # VAE-style context encoder layers
        # Input Layer
        self.ce_in = nn.Linear(context_input_dim, context_layer_sizes[0])
        # Hidden Layers
        self.ce_h1 = nn.Linear(context_layer_sizes[0], context_layer_sizes[1])
        self.ce_h2 = nn.Linear(context_layer_sizes[1], context_layer_sizes[2])
        # Output Layers
        self.ce_out_mean = nn.Linear(context_layer_sizes[2], context_latent_size)
        self.ce_out_var = nn.Linear(context_layer_sizes[2], context_latent_size)

        self.ce_velo_mean = nn.Linear(context_layer_sizes[2], context_torso_velo_size)
        self.ce_velo_var  = nn.Linear(context_layer_sizes[2], context_torso_velo_size)

        # self.ce_timestep = nn.Linear(context_layer_sizes[2], 1)

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.drop_3 = nn.Dropout(dropout)

        self.device = device
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.ce_in, self.ce_h1, self.ce_h2,
                     self.ce_out_mean, self.ce_out_var, 
                     self.ce_velo_mean, self.ce_velo_var]:
            
            nn.init.xavier_uniform_(layer.weight)
            
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def encode(self, X_C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through encoder
        x = F.elu(self.ce_in(X_C))
        x = self.drop_1(x)
        x = F.elu(self.ce_h1(x))
        x = self.drop_2(x)
        x = F.elu(self.ce_h2(x))
        x = self.drop_3(x)

        return self.ce_out_mean(x), self.ce_out_var(x), self.ce_velo_mean(x), self.ce_velo_var(x)

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
        mean, logvar, v_mean, v_logvar = self.encode(X_C)
        z = self.reparameterization_trick(mean, logvar)
        torso_velo = self.reparameterization_trick(v_mean, v_logvar)
        return mean, logvar, z, torso_velo

class ContextDecoder(nn.Module):
    """Decoder network for reconstructing next state from latent representation and velocity.
    
    Takes a latent vector and torso velocity as input, processes through two ELU-activated
    hidden layers, and outputs a predicted next state. Uses Xavier uniform initialization.
    """

    def __init__(
            self,
            input_dim: int = 32,
            layers: List[int] = [64,128,256,128,92],
            decode_dim: int = 82,
            dropout: float = 0.1) -> None:
        super().__init__()


        # Network architecture
        self.dec_in = nn.Linear(input_dim, layers[0])
        self.dec_h1 = nn.Linear(layers[0], layers[1])
        self.dec_h2 = nn.Linear(layers[1], layers[2])
        self.dec_h3 = nn.Linear(layers[2], layers[3])
        self.dec_h4 = nn.Linear(layers[3], layers[4])
        self.dec_out = nn.Linear(layers[4], decode_dim)

        self.drop_1 = nn.Dropout(dropout)
        self.drop_2 = nn.Dropout(dropout)
        self.drop_3 = nn.Dropout(dropout)
        self.drop_4 = nn.Dropout(dropout)
        self.drop_5 = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize all linear layers with Xavier uniform distribution."""
        for layer in [self.dec_in, self.dec_h1, self.dec_h2, self.dec_h3, self.dec_h4, self.dec_out]:
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
        x = F.elu(self.dec_h2(x))
        x = self.drop_3(x)
        x = F.elu(self.dec_h3(x))
        x = self.drop_4(x)
        x = F.elu(self.dec_h4(x))
        x = self.drop_5(x)
        return self.dec_out(x)


class ActorCritic_Dynamic(nn.Module):
    def __init__(self, 
                 num_actor_obs=69, 
                 num_critic_obs=95, 
                 num_actions=12,
                 actor_shared_dim=512,
                 actor_branch_layers=[256,128,64],
                 cenet_in_dim=350,
                #  cenet_out_dim=88,
                 cenet_latent_dim=29,
                 cenet_velo_dim=3, 
                 cenet_enc_layers=[256,128,64],
                #  cenet_dec_layers=[64,128,256,128,92],
                 dropout=0.1,
                 activation="elu", 
                 init_noise_std=1.0,):
        super().__init__()

        # Construct the context encoder network
        self.context_encoder = ContextEncoder(context_input_dim=cenet_in_dim,
                                              context_layer_sizes=cenet_enc_layers,
                                              context_latent_size=cenet_latent_dim,
                                              context_torso_velo_size=cenet_velo_dim,
                                              dropout=dropout)
        
        # # Construct the context decoder network
        # self.context_decoder = ContextDecoder(input_dim=cenet_latent_dim+cenet_velo_dim,
        #                                       decode_dim=cenet_out_dim,
        #                                       layers=cenet_dec_layers,
        #                                       dropout=dropout)
        
        # Get the activation function used by the actor and critic networks
        self.activation = get_activation(activation)

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
        self.act_pos_h3  = nn.Linear(actor_branch_layers[1], actor_branch_layers[2])
        self.act_pos_out = nn.Linear(actor_branch_layers[2], num_actions)

        #     Branch for torque control
        self.act_tau_h1 = nn.Linear(actor_shared_dim, actor_branch_layers[0])
        self.act_tau_h2  = nn.Linear(actor_branch_layers[0], actor_branch_layers[1])
        self.act_tau_h3  = nn.Linear(actor_branch_layers[1], actor_branch_layers[2])
        self.act_tau_out = nn.Linear(actor_branch_layers[2], num_actions)

        # Now create FiLM layers for sharing info. between branches
        #     Applied after h1
        self.act_pos_2_tau_h1 = _ShiftScaleMod(dim=actor_branch_layers[0])
        self.act_tau_2_pos_h1 = _ShiftScaleMod(dim=actor_branch_layers[0])
        #     Applied after h2
        self.act_pos_2_tau_h2 = _ShiftScaleMod(dim=actor_branch_layers[1])
        self.act_tau_2_pos_h2 = _ShiftScaleMod(dim=actor_branch_layers[1])
        #     Applied after h3
        self.act_pos_2_tau_h3 = _ShiftScaleMod(dim=actor_branch_layers[2])
        self.act_tau_2_pos_h3 = _ShiftScaleMod(dim=actor_branch_layers[2])

        # Dropout layers...
        self.shared_drop = nn.Dropout(p=dropout)
        self.h1_pos_drop = nn.Dropout(p=dropout)
        self.h1_tau_drop = nn.Dropout(p=dropout)
        self.h2_pos_drop = nn.Dropout(p=dropout)
        self.h2_tau_drop = nn.Dropout(p=dropout)
        self.h3_pos_drop = nn.Dropout(p=dropout)
        self.h3_tau_drop = nn.Dropout(p=dropout)

        ###
        #  Construct layers for the critic network
        ###
        #     critic layers...
        self.critic_in  = nn.Linear(num_critic_obs, actor_shared_dim)
        self.critic_h1  = nn.Linear(actor_shared_dim, actor_branch_layers[0])
        self.critic_h2  = nn.Linear(actor_branch_layers[0], actor_branch_layers[1])
        self.critic_h3  = nn.Linear(actor_branch_layers[1], actor_branch_layers[2])
        self.critic_out = nn.Linear(actor_branch_layers[2], 1)

        #     dropout layers...
        self.critic_in_drop = nn.Dropout(p=dropout)
        self.critic_h1_drop = nn.Dropout(p=dropout)
        self.critic_h2_drop = nn.Dropout(p=dropout)
        self.critic_h3_drop = nn.Dropout(p=dropout)

        self.critic = nn.Sequential(
            self.critic_in,
            self.activation,
            self.critic_in_drop,
            self.critic_h1,
            self.activation,
            self.critic_h1_drop,
            self.critic_h2,
            self.activation,
            self.critic_h2_drop,
            self.critic_h3,
            self.activation,
            self.critic_h3_drop,
            self.critic_out
        )

        # Used to track these values during training....
        #     These values will not be used during inference (sim or real)
        self.cenet_mean = None 
        self.cenet_logvar = None
        self.cenet_z = None
        self.cenet_torso_velo = None

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def _init_weights(self):
        # Shared input encoding layer
        nn.init.xavier_uniform_(self.actor_shared_input.weight)
        # Actor position-control layers
        nn.init.xavier_uniform_(self.act_pos_h1.weight)
        nn.init.xavier_uniform_(self.act_pos_h2.weight)
        nn.init.xavier_uniform_(self.act_pos_h3.weight)
        nn.init.xavier_uniform_(self.act_pos_out.weight)
        # Actor torque control layers
        nn.init.xavier_uniform_(self.act_tau_h1.weight)
        nn.init.xavier_uniform_(self.act_tau_h2.weight)
        nn.init.xavier_uniform_(self.act_tau_h3.weight)
        nn.init.xavier_uniform_(self.act_tau_out.weight)
        # Critic layers
        nn.init.xavier_uniform_(self.critic_in.weight)
        nn.init.xavier_uniform_(self.critic_h1.weight)
        nn.init.xavier_uniform_(self.critic_h2.weight)
        nn.init.xavier_uniform_(self.critic_h3.weight)
        nn.init.xavier_uniform_(self.critic_out.weight)

        # Basis if they exists
        #     Shared
        if self.actor_shared_input.bias is not None:
            nn.init.zeros_(self.actor_shared_input.bias)
        #     Actor position
        if self.act_pos_h1.bias is not None:
            nn.init.zeros_(self.act_pos_h1.bias)
        if self.act_pos_h2.bias is not None:
            nn.init.zeros_(self.act_pos_h2.bias)
        if self.act_pos_h3.bias is not None:
            nn.init.zeros_(self.act_pos_h3.bias)
        if self.act_pos_out.bias is not None:
            nn.init.zeros_(self.act_pos_out.bias)
        #     Actor torque
        if self.act_tau_h1.bias is not None:
            nn.init.zeros_(self.act_tau_h1.bias)
        if self.act_tau_h2.bias is not None:
            nn.init.zeros_(self.act_tau_h2.bias)
        if self.act_tau_h3.bias is not None:
            nn.init.zeros_(self.act_tau_h3.bias)
        if self.act_tau_out.bias is not None:
            nn.init.zeros_(self.act_tau_out.bias)
        #     Critic layers
        if self.critic_in.bias is not None:
            nn.init.zeros_(self.critic_in.bias)
        if self.critic_h1.bias is not None:
            nn.init.zeros_(self.critic_h1.bias)
        if self.critic_h2.bias is not None:
            nn.init.zeros_(self.critic_h2.bias)
        if self.critic_h3.bias is not None:
            nn.init.zeros_(self.critic_h3.bias)
        if self.critic_out.bias is not None:
            nn.init.zeros_(self.critic_out.bias)
        

    def get_optim_groups(self, weight_decay: float = 1e-3, strong_decay: float = 1e-1):
        """Separate parameters into groups with and without weight decay.
        
        Args:
            weight_decay (float): Weight decay value for regularization. Default: 1e-3.
            
        Returns:
            List of parameter groups for optimizer initialization.
        """
        decay     = set()
        no_decay  = set()
        special_decay = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight"):
                    if isinstance(m, whitelist):
                        if "_2_" in fpn:
                            # Here is the name contains a "2" then it is a cross-conditioning
                            #    FILM layer, and we want a stronger weight reg on these values
                            special_decay.add(fpn)
                        else:
                            decay.add(fpn)
                    elif isinstance(m, blacklist):
                        no_decay.add(fpn)

        # for i in range(self.options["action_net"]["num_layers"]-1):
        #     no_decay.update([f"noise_decoder.cross_field_scales_pos.{i}", f"noise_decoder.cross_field_scales_tau.{i}"])

        # Validate parameter separation
        param_dict   = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay & special_decay
        if inter_params:
            raise ValueError(f"Parameters in all sets: {inter_params}")
        missing_params = param_dict.keys() - (decay | no_decay | special_decay)
        if missing_params:
            raise ValueError(f"Parameters not categorized: {missing_params}")
        
        print(f"Parameters with extra strong weight decay{special_decay}")

        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(special_decay)], "weight_decay": strong_decay}
        ]

    def configure_optimizers(self,
                           learning_rate: float = 1e-4,
                           weight_decay: float = 1e-3,
                           strong_decay: float = 1e-1,
                           betas: Tuple[float, float] = (0.9, 0.999)) -> torch.optim.Optimizer:
        """Configure the AdamW optimizer with parameter groups.

        Standard weights in Linear/Attention layers - weight_decay
        all bias terms - no decay
        FiLM layer parameters - strong_decay
            
        Returns:
            Configured AdamW optimizer.
        """
        optim_groups = self.get_optim_groups(weight_decay=weight_decay, strong_decay=strong_decay)
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    # forward methods for the histroical context VAE
    def cenet_enc_forward(self, obs_history):
        mean, logvar, z, torso_velo = self.context_encoder(obs_history)
        return mean, logvar, z, torso_velo
    
    # def cenet_dec_forward(self, history_latent):
    #     # This is assuming all the elements used as input to the forward method
    #     #     of the decoder network have been concatonated before being passed
    #     #     to this function!
    #     next_obs = self.context_decoder(history_latent)
    #     return next_obs

    # Method for the forward method of the actor network, used mostly as an internal method
    def actor_forward(self, current_obs):
        # run the concatonated vector through the shared encoder layer
        x = self.actor_shared_input(current_obs)
        x = self.activation(x)
        x = self.shared_drop(x)

        # Now run the two parallel branches
        #     position
        pos_latent = self.act_pos_h1(x)
        pos_latent = self.activation(pos_latent)
        #     torque
        tau_latent = self.act_tau_h1(x)
        tau_latent = self.activation(tau_latent)
        #     now perform the cross-conditioning
        pos_latent = self.act_tau_2_pos_h1(pos_latent, tau_latent)  # perform FiLM on pos_latent using tau_latent
        tau_latent = self.act_pos_2_tau_h1(tau_latent, pos_latent)  # perform FiLM on tau_latent using pos_latent
        #     dropout AFTER sharing
        pos_latent = self.h1_pos_drop(pos_latent)
        tau_latent = self.h1_tau_drop(tau_latent)

        # REPEAT
        #     position
        pos_latent = self.act_pos_h2(pos_latent)
        pos_latent = self.activation(pos_latent)
        #     torque
        tau_latent = self.act_tau_h2(tau_latent)
        tau_latent = self.activation(tau_latent)
        #     now perform the cross-conditioning
        pos_latent = self.act_tau_2_pos_h2(pos_latent, tau_latent)  # perform FiLM on pos_latent using tau_latent
        tau_latent = self.act_pos_2_tau_h2(tau_latent, pos_latent)  # perform FiLM on tau_latent using pos_latent
        #     dropout AFTER sharing
        pos_latent = self.h2_pos_drop(pos_latent)
        tau_latent = self.h2_tau_drop(tau_latent)    
    
        #     position
        pos_latent = self.act_pos_h3(pos_latent)
        pos_latent = self.activation(pos_latent)
        #     torque
        tau_latent = self.act_tau_h3(tau_latent)
        tau_latent = self.activation(tau_latent)
        #     now perform the cross-conditioning
        pos_latent = self.act_tau_2_pos_h3(pos_latent, tau_latent)  # perform FiLM on pos_latent using tau_latent
        tau_latent = self.act_pos_2_tau_h3(tau_latent, pos_latent)  # perform FiLM on tau_latent using pos_latent
        #     dropout AFTER sharing
        pos_latent = self.h3_pos_drop(pos_latent)
        tau_latent = self.h3_tau_drop(tau_latent)

        # Now run the final output layers to get both action modalities
        act_pos_act = self.act_pos_out(pos_latent)
        act_tau_act = self.act_tau_out(tau_latent)

        out = torch.cat([act_pos_act, act_tau_act], dim=-1)

        return out
    
    # Functions that are specific to PPO training
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_distribution(self, curr_obs):
        mean = self.actor_forward(curr_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    # method used during simulated training
    def act(self, obs, obs_history, **kwargs):
        # Call the forward method of the context encoder
        mean, logvar, z, torso_velo = self.cenet_dec_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # Upated the PPO training distribution
        self.update_distribution(current_obs)
        
        # log context-encoder values to be used in PPO class for calculating context encoder network
        self.cenet_mean = mean
        self.cenet_logvar = logvar
        self.cenet_z = z
        self.cenet_torso_velo = torso_velo
        
        # return a sample from the distribution to be executed in simulation
        return self.distribution.sample()

    # Method using during simulated inference
    def act_inference(self,obs,obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo = self.cenet_dec_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_mean = self.actor_forward(current_obs)
        return actions_mean
    
    # Method to run inference on hardware WITHOUT logging the VAE's outputs
    @torch.jit.export
    def act_inference_deploy(self, obs, obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo = self.cenet_dec_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_mean = self.actor_forward(current_obs)
        return actions_mean
    
    @torch.jit.export
    def act_inference_deploy_log(self, obs, obs_history):
        # Call the forward method of the context encoder
        _, _, z, torso_velo = self.cenet_dec_forward(obs_history)
        
        # create the actors observation
        current_obs = torch.cat((obs,z,torso_velo), dim=-1)   
        
        # call the actors forward method and return it's results
        actions_mean = self.actor_forward(current_obs)
        return actions_mean, z, torso_velo


    # Forward method for calculating the value of the current state
    #     using the privilged critic observation
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


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
    else:
        print("invalid activation function!")
        return None
