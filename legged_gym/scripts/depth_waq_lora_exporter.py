import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.scripts.joystick import Joystick
from legged_gym.utils.exp_data_logger import ExpLogger

import numpy as np
import torch
from rsl_rl.utils.LoRA import SequentialMultiLora, MultiLora
import copy

import argparse

import cv2

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class DepthCNNExporter(torch.nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()

        # Reuse the already-constructed LoRA swapper.
        self.visual_encoder = visual_encoder

    @torch.jit.export
    def forward(self, depth_image):
        return self.visual_encoder(depth_image)

    @torch.jit.export
    def swap(self, index: int):
        self.visual_encoder.swap(index)

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)


class PolicyExporterFeaturesWaQ(torch.nn.Module):
    @torch.jit.unused
    def __init__(self, actor, vae_encoder, vae_latent_mu, vae_vel_mu, num_of_loras):
        super().__init__()
        # Already constructed LoRA swappers
        self.actor = actor

        self.vae_encoder = vae_encoder
        self.vae_latent_mu = vae_latent_mu
        self.vae_vel_mu = vae_vel_mu

        self.num_of_loras = num_of_loras

    def vae_inference(self, obs_history):
        encoded = self.vae_encoder(obs_history)
        latent_mu = self.vae_latent_mu(encoded)
        vel_mu = self.vae_vel_mu(encoded)
        return torch.cat((latent_mu, vel_mu), dim=-1)

    @torch.jit.export
    def forward(self, observations, obs_history, visual_latent):
        mean_out = self.vae_inference(obs_history)
        all_obs = torch.cat((observations, mean_out, visual_latent), dim=-1,)
        return self.actor(all_obs)

    @torch.jit.export
    def swap(self, index: int):
        self.actor.swap(index)
        self.vae_encoder.swap(index)
        self.vae_latent_mu.swap(index)
        self.vae_vel_mu.swap(index)

    @torch.jit.unused
    def export(self, filename):
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)



class PolicyExporterDepthWaQ(torch.nn.Module):
    @torch.jit.unused
    def __init__(self, actor_critic):
        super().__init__()
        actor_critic = copy.deepcopy(actor_critic)
        actor = actor_critic.actor
        vae = actor_critic.vae
        visual_encoder = actor_critic.visual_encoder

        #actor
        self.actor = SequentialMultiLora(actor)

        #vae
        self.vae_encoder = SequentialMultiLora(vae.encoder)
        self.vae_latent_mu = MultiLora(vae.latent_mu)
        self.vae_vel_mu = MultiLora(vae.vel_mu)

        #visual encoder
        self.visual_encoder = SequentialMultiLora(visual_encoder.cnn)

        self.num_of_loras = 0

    @torch.jit.unused
    def append(self, actor_critic_lora):
        actor_critic_lora = copy.deepcopy(actor_critic_lora)
        actor = actor_critic_lora.actor
        vae = actor_critic_lora.vae
        visual_encoder = actor_critic_lora.visual_encoder

        self.actor.append(actor)
        self.vae_encoder.append(vae.encoder)
        self.vae_latent_mu.append(vae.latent_mu)
        self.vae_vel_mu.append(vae.vel_mu)

        self.visual_encoder.append(visual_encoder.cnn)

        self.num_of_loras += 1
    
    @torch.jit.unused
    def export(self, filename):
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(filename)

    def depth_actor(self, observations, samples, depth_image):
        visual_latent = self.visual_encoder(depth_image)
        return self.actor(torch.cat((observations, samples, visual_latent), dim=-1))

    def vae_inference(self, obs_history):
        encoded = self.vae_encoder(obs_history)
        latent_mu = self.vae_latent_mu(encoded)
        vel_mu = self.vae_vel_mu(encoded)
        return torch.cat((latent_mu, vel_mu), dim=-1)
    
    @torch.jit.export
    def forward(self, observations, obs_history, depth_image):
        mean_out = self.vae_inference(obs_history)
        return self.depth_actor(observations, mean_out, depth_image)

    @torch.jit.export
    def swap(self, index: int):
        self.actor.swap(index)
        self.vae_encoder.swap(index)
        self.vae_latent_mu.swap(index)
        self.vae_vel_mu.swap(index)
        self.visual_encoder.swap(index)
    

    @torch.jit.unused
    def split_cnn(self):
        """
        Split the already-constructed CNN LoRA swapper from the policy.
        No new LoRA swappers are constructed.
        """
        cnn = DepthCNNExporter(self.visual_encoder)
        main = PolicyExporterFeaturesWaQ(self.actor, self.vae_encoder, self.vae_latent_mu, self.vae_vel_mu, self.num_of_loras)
        return cnn, main

import os
import copy
import torch


def _iter_seq(seq):
    """Works for both eager nn.Sequential and a scripted ScriptModule wrapping one."""
    for name, child in seq.named_children():
        yield int(name), child

def _collect_base_weights(policy):
    """Grab a detached snapshot of every base_module.weight the swap() touches.
    Works on both the eager nn.Module and a loaded torch.jit.ScriptModule —
    attribute access is the same for both."""
    weights = {}
    for name in ["actor", "vae_encoder", "visual_encoder"]:
        seq = getattr(policy, name)
        for i, layer in _iter_seq(seq):
            # isinstance check against MultiLora won't work reliably on a
            # ScriptModule (its __class__ is a mangled TorchScript type),
            # so fall back to duck-typing via hasattr
            if hasattr(layer, "weight"):
                weights[f"{name}.{i}"] = layer.weight.detach().clone()

    for name in ["vae_latent_mu", "vae_vel_mu"]:
        layer = getattr(policy, name)
        weights[name] = layer.weight.detach().clone()

    return weights


def _assert_swap_behaves(policy, label):
    base_weights = _collect_base_weights(policy)

    policy.swap(0)
    swapped_weights = _collect_base_weights(policy)

    changed_any = False
    for key in base_weights:
        before, after = base_weights[key], swapped_weights[key]
        assert before.shape == after.shape, f"[{label}] {key}: shape changed unexpectedly"
        if not torch.equal(before, after):
            changed_any = True
        else:
            print(f"[{label}] WARNING: {key} did not change after swap(0)")

    assert changed_any, f"[{label}] swap(0) did not change ANY weight"

    policy.swap(-1)
    restored_weights = _collect_base_weights(policy)

    for key in base_weights:
        before, restored = base_weights[key], restored_weights[key]
        assert torch.allclose(before, restored, atol=1e-6), (
            f"[{label}] {key}: swap(-1) did not restore original weights "
            f"(max diff = {(before - restored).abs().max().item()})"
        )

    print(f"[{label}] swap() correctly modifies (if no errors) and restores weights.")


def test_swap_eager_and_exported(policy, tmp_export_dir):
    """
    policy: PolicyExporterDepthWaQ with >=1 lora already appended
    tmp_export_dir: scratch directory to write the exported .pt into
    """
    # --- 1. eager module, pre-export ---
    _assert_swap_behaves(policy, label="eager")

    # sanity: eager test must leave current_index back at -1, otherwise the
    # export below would ship a policy with an active lora baked in
    for name in ["actor", "vae_encoder", "visual_encoder"]:
        for layer in getattr(policy, name):
            if hasattr(layer, "current_index"):
                assert layer.current_index == -1, f"{name} left with lora still active"
    for name in ["vae_latent_mu", "vae_vel_mu"]:
        assert getattr(policy, name).current_index == -1

    # --- 2. export, then reload the actual saved artifact ---
    policy.export(tmp_export_dir)
    saved_files = [f for f in os.listdir(tmp_export_dir) if f.endswith(".pt")]
    assert saved_files, "export() did not produce a .pt file"
    export_path = os.path.join(tmp_export_dir, sorted(saved_files)[-1])  # most recent

    loaded = torch.jit.load(export_path, map_location="cpu")

    # --- 3. run the same swap test on the loaded, scripted module ---
    _assert_swap_behaves(loaded, label="exported+reloaded")

    # --- 4. explicitly confirm loras tensors survived export on the right device ---
    #for name in ["actor", "vae_encoder", "visual_encoder"]:
    #    for layer in getattr(loaded, name):
    #        if hasattr(layer, "loras"):
    #            for (lora_A, lora_B, _scale) in layer.loras:
    #                assert lora_A.device.type == "cpu", (
    #                    f"{name}: lora_A stayed on {lora_A.device} after export/reload "
    #                    "(likely the loras-not-following-.to() bug)"
    #                )
    #                assert lora_B.device.type == "cpu"


# usage:
# policy = PolicyExporterDepthWaQ(actor_critic)
# policy.append(actor_critic_lora)
# test_swap_eager_and_exported(policy, tmp_export_dir="/tmp/lora_export_test")
    

def loader(actor_critic, file):
    model_dict, args = torch.load(file[0])["model_state_dict"], torch.load(file[1])["args"]
    print(args)
    model = actor_critic(**args)
    model.load_state_dict(model_dict)
    return model

if __name__ == "__main__":
    from rsl_rl.modules import ActorCriticDreamWaQDepth, ActorCriticDreamWaQDepthLora


    
    base_model = loader(
        ActorCriticDreamWaQDepth, 
        (f"/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_baseline/Aug09_02-33-11_dreamwaq_isaacgym/model_7000.pt",f"/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_baseline/Aug09_02-33-11_dreamwaq_isaacgym/current_actor_args.pt")
        
    )

    lora_raw = [
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_gap/Aug10_18-19-15_dreamwaq_isaacgym", 40000),
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_all_stairs/Aug10_22-06-45_dreamwaq_isaacgym", 40000),
        (f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_pit/Aug17_13-44-54_dreamwaq_isaacgym", 60000)
        #(f"{LEGGED_GYM_ROOT_DIR}/logs/go2_depth_waq_lora_8_pit_experiment1_better_headings/Aug04_23-58-15_dreamwaq_isaacgym", 25000),
    ]

    #loras_files = [
    #    (/.pt", "/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_lora_8_gap_experiment1_better_headings/Aug01_08-38-05_dreamwaq_isaacgym/"),
    #    ("/model_pt","/home/pablo/Documents/Legged_Gym_EX/logs/go2_depth_waq_lora_8_stairs_experiment1_better_headings/Jul30_23-25-59_dreamwaq_isaacgym/current_actor_args.pt")
    #
    #]

    loras_files = [
        (os.path.join(folder, f"model_{model}.pt"), os.path.join(folder, f"current_actor_args.pt"))
        for folder, model in lora_raw

    ]

    loras = [
        loader(ActorCriticDreamWaQDepthLora, file)
        for file in loras_files
    ]

    exporter = PolicyExporterDepthWaQ(base_model)
    for x in loras:
        exporter.append(x)

    #print(exporter)
    #observations=torch.randn(1, 45)
    #obs_history=torch.randn(1, 900)
    #depth_image=torch.randn(1, 1, 48, 64)
    #output = exporter(observations, obs_history, depth_image)


    #test_swap_eager_and_exported(exporter, tmp_export_dir="/tmp/lora_export_test")

    path = os.path.join(LEGGED_GYM_ROOT_DIR, "exported")
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, timestamp)
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, f"compiled_lora_{timestamp}.pt")
    exporter.export(file)
    cnn, main = exporter.split_cnn()
    path = os.path.join(path, f"compiled_lora_{timestamp}_split")
    os.makedirs(path, exist_ok=True)
    cnn.export(os.path.join(path, f"DepthCNN.pt"))
    main.export(os.path.join(path, f"FeaturesWaQ.pt"))
    
    #policy = torch.jit.load(os.path.join(path, f"compiled_lora_{timestamp}.pt"))
    #print(
    #    torch.sum(output - policy(observations, obs_history, depth_image))
    #)
            