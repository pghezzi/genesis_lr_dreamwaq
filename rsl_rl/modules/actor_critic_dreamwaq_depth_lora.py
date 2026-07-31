import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


from rsl_rl.modules.vae import VAE
from rsl_rl.modules.actor_critic_dreamwaq_depth import ActorCriticDreamWaQDepth
from rsl_rl.utils import LoRA

'''
Actor-Critic for Hybrid Implicit-Explicit architecture using VAE
'''

class ActorCriticDreamWaQDepthLora(ActorCriticDreamWaQDepth):
    is_recurrent = False

    def __init__(
        self,
        *args,
        base_model= None,
        actor_ranks = 4,
        encoder_ranks = 4,
        decoder_ranks = 4,
        latent_mu_rank= 4,
        vel_mu_rank = 4,
        latent_var_ranks = 4,
        vel_var_ranks = 4,
        visual_encoder_ranks = 4,
        **kwargs
    ):
        print("BASE_MODEL")
        super().__init__(*args, **kwargs)

        # Apply LoRA
        self.actor = LoRA._from_sequential(self.actor, actor_ranks)
        self.vae.encoder = LoRA._from_sequential(self.vae.encoder, encoder_ranks)

        self.vae.latent_mu = LoRA.LoRALinear._from_linear(
            self.vae.latent_mu, latent_mu_rank
        )
        self.vae.vel_mu = LoRA.LoRALinear._from_linear(
            self.vae.vel_mu, vel_mu_rank
        )

        self.vae.latent_var = LoRA._from_sequential(
            self.vae.latent_var, latent_var_ranks
        )
        self.vae.vel_var = LoRA._from_sequential(
            self.vae.vel_var, vel_var_ranks
        )

        self.vae.decoder = LoRA._from_sequential(
            self.vae.decoder, decoder_ranks
        )

        self.visual_encoder.cnn = LoRA._from_sequential(
            self.visual_encoder.cnn, visual_encoder_ranks
        )

        # Optional base model load
        #if base_model is not None:
        assert base_model is not None
        print(f"Loading baseline: {base_model}")
        loaded_dict = torch.load(base_model)
        self.load_state_dict(loaded_dict["model_state_dict"])

        print("OVERRIDE WITH LORA")
        print(f"Encoder MLP (LORA): {self.vae.encoder}")
        print(f"Decoder MLP (LORA): {self.vae.decoder}")
        print(f"Actor MLP (LORA): {self.actor}")
        print(f"Visual Encoder (LORA): {self.visual_encoder.cnn}")

    def load_state_dict(self, *args, **kwargs):
        super().load_state_dict(*args, **kwargs, strict=False)



if __name__ == "__main__":
    from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg
    from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfgPPO

    cfg = Go2DreamwaqCfg()
    cfgppo = Go2DreamwaqCfgPPO()
    ActorCriticDreamWaQLoRA(
                num_actor_obs = cfg.env.num_observations,
                num_actions=cfg.env.num_actions,
                num_privileged_obs=cfg.env.num_privileged_obs, 
                num_history_input=cfg.env.num_history_obs,
                num_latent_dims=cfg.env.num_latent_dims,
                num_explicit_dims=cfg.env.num_explicit_dims,
                num_decoder_output=cfg.env.num_decoder_output,
                #actor_hidden_dims=cfgppo.policy.,
                critic_hidden_dims=cfgppo.policy.critic_hidden_dims,
                encoder_hidden_dims=cfgppo.policy.encoder_hidden_dims,
                decoder_hidden_dims=cfgppo.policy.decoder_hidden_dims,
                activation='elu',
                init_noise_std=1.0,
    )