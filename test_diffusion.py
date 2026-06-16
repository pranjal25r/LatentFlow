#!/usr/bin/env python3
"""Smoke test for diffusion components."""

import sys
import torch

sys.path.insert(0, '/Users/pranjal/Desktop/Home/Projects/LatentFlow')

from latentflow.diffusion import GaussianDiffusion
from latentflow.models.dit import DiT_S_2

print("=" * 80)
print("Gaussian Diffusion Smoke Test")
print("=" * 80)

# Initialize diffusion
diffusion = GaussianDiffusion(
    num_timesteps=1000,
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
    prediction_type="epsilon",
)
print(f"\n✓ Created GaussianDiffusion")
print(f"  Timesteps: {diffusion.num_timesteps}")
print(f"  Beta schedule: linear")

# Create a small DiT model
model = DiT_S_2(latent_channels=4, dropout=0.0)
print(f"\n✓ Created DiT-S/2 model")
print(f"  Parameters: {model.num_parameters():,}")

# Test forward diffusion (q_sample)
batch_size = 2
latent_channels = 4
latent_h = latent_w = 32

x0 = torch.randn(batch_size, latent_channels, latent_h, latent_w)
timesteps = torch.tensor([100, 500])

x_t = diffusion.q_sample(x0, timesteps)
print(f"\n✓ Forward diffusion (q_sample)")
print(f"  x0 shape: {tuple(x0.shape)}")
print(f"  x_t shape: {tuple(x_t.shape)}")
assert x_t.shape == x0.shape, f"Shape mismatch: {x_t.shape} != {x0.shape}"

# Test training loss
loss = diffusion.training_loss(model, x0, timesteps)
print(f"\n✓ Training loss")
print(f"  Loss: {loss.item():.4f}")
assert loss.dim() == 0 and loss.item() > 0, "Loss should be scalar and positive"

# Test single reverse step (p_sample)
print(f"\n✓ Single denoising step (p_sample)")
x_prev = diffusion.p_sample(model, x_t, t=100)
print(f"  x_t shape: {tuple(x_t.shape)}")
print(f"  x_{{t-1}} shape: {tuple(x_prev.shape)}")
assert x_prev.shape == x_t.shape, f"Shape mismatch in p_sample"

# Test DDIM sampling (subset of timesteps)
print(f"\n" + "=" * 80)
print("Testing DDIM Sampling (fast, deterministic)")
print("=" * 80)

shape = (1, 4, 32, 32)  # Single sample for speed
num_ddim_steps = 10  # Fast test

with torch.no_grad():
    samples = diffusion.ddim_sample(
        model,
        shape=shape,
        num_steps=num_ddim_steps,
        guidance_scale=1.0,
        eta=0.0,  # Deterministic
    )

print(f"\n✓ DDIM sampling completed")
print(f"  Generated shape: {tuple(samples.shape)}")
print(f"  DDIM steps: {num_ddim_steps}")
assert samples.shape == shape, f"Shape mismatch in DDIM: {samples.shape} != {shape}"

# Test DDPM sampling (full timesteps) with just a few steps for speed
print(f"\n" + "=" * 80)
print("Testing DDPM Sampling (full process, stochastic)")
print("=" * 80)

# Create smaller diffusion for speed
diffusion_small = GaussianDiffusion(
    num_timesteps=100,  # Fewer steps for speed
    beta_schedule="linear",
    beta_start=0.0001,
    beta_end=0.02,
)

shape = (1, 4, 32, 32)
with torch.no_grad():
    samples_ddpm = diffusion_small.p_sample_loop(
        model,
        shape=shape,
        guidance_scale=1.0,
    )

print(f"\n✓ DDPM sampling completed")
print(f"  Generated shape: {tuple(samples_ddpm.shape)}")
print(f"  DDPM steps: {diffusion_small.num_timesteps}")
assert samples_ddpm.shape == shape, f"Shape mismatch in DDPM"

print(f"\n" + "=" * 80)
print("All diffusion tests passed! ✓")
print("=" * 80)
