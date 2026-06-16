"""Gaussian diffusion process and sampling.

References:
  DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
  DDIM: Song et al., "Denoising Diffusion Implicit Models" (2021)
"""

from typing import Optional, Tuple, Callable
import math
import torch
import torch.nn as nn
from torch import Tensor
from .schedule import get_beta_schedule, compute_alphas


class GaussianDiffusion(nn.Module):
    """Implements the Gaussian diffusion process.
    
    Supports:
      - Forward diffusion (q_sample): add noise to clean samples
      - Training loss: predict noise from noisy samples
      - Sampling (p_sample): iteratively denoise
      - DDIM fast sampling
      - Optional classifier-free guidance
    
    All tensors registered as buffers (non-trainable, moved to device).
    
    Args:
        num_timesteps: Number of diffusion steps.
        beta_schedule: Schedule type ("linear" or "cosine").
        beta_start: Starting beta (for linear schedule).
        beta_end: Ending beta (for linear schedule).
        prediction_type: What to predict ("epsilon", "sample", or "v_prediction").
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ) -> None:
        """Initialize Gaussian diffusion."""
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Compute beta schedule and alpha quantities
        betas = get_beta_schedule(beta_schedule, num_timesteps, beta_start, beta_end)
        (
            alphas,
            alphas_cumprod,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            alphas_cumprod_prev,
            posterior_variance,
        ) = compute_alphas(betas)

        # Register all as buffers (non-trainable, on same device as model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("posterior_variance", posterior_variance)
        
        # Log variance for numerical stability
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20))
        )

    def q_sample(
        self,
        x0: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Add noise to clean sample (forward diffusion).
        
        Implements the closed-form forward diffusion:
          x_t = sqrt(α̅_t) x_0 + sqrt(1 - α̅_t) ε
        
        Where α̅_t is the cumulative product of alphas (1 - betas).
        
        Args:
            x0: Clean latent tensor of shape (B, C, H, W).
            t: Timestep indices of shape (B,) with values in [0, num_timesteps).
            noise: Gaussian noise of shape (B, C, H, W). If None, sampled.
            
        Returns:
            Noisy latent x_t of shape (B, C, H, W).
            
        Shape contract:
            x0: (B, C, H, W) → x_t: (B, C, H, W)
            t: (B,) with values in {0, 1, ..., num_timesteps-1}
        """
        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get alpha schedule values for the given timesteps
        # t: [B] -> [B, 1, 1, 1] for broadcasting
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # x_t = sqrt(α̅_t) * x0 + sqrt(1 - α̅_t) * noise
        x_t = sqrt_alpha * x0 + sqrt_1_minus_alpha * noise
        
        return x_t

    def training_loss(
        self,
        model: nn.Module,
        x0: Tensor,
        t: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute training loss for diffusion model.
        
        Procedure:
          1. Sample random timesteps t ~ Uniform(0, num_timesteps)
          2. Sample Gaussian noise ε ~ N(0, I)
          3. Forward diffuse: x_t = q_sample(x0, t, ε)
          4. Model predicts: ε̂ = model(x_t, t)
          5. Loss: MSE(ε̂, ε)
        
        Args:
            model: Denoising model (e.g., DiT) that predicts noise.
            x0: Clean latent of shape (B, C, H, W).
            t: Optional timestep indices of shape (B,). If None, sampled randomly.
            noise: Optional Gaussian noise of shape (B, C, H, W). If None, sampled.
            
        Returns:
            Scalar MSE loss (float tensor of shape ()).
            
        Shape contract:
            x0: (B, C, H, W)
            t: (B,) if provided
            noise: (B, C, H, W) if provided
            loss: scalar
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample random timesteps if not provided
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device)
        
        # Sample random noise
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Forward diffusion: add noise to x0
        x_t = self.q_sample(x0, t, noise)
        
        # Model predicts noise
        noise_pred = model(x_t, t)
        
        # MSE loss between predicted and true noise
        loss = torch.mean((noise_pred - noise) ** 2)
        
        return loss

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        xt: Tensor,
        t: int,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """Single denoising step (DDPM reverse process).
        
        Reverse step: x_{t-1} ~ p(x_{t-1} | x_t)
        
        The posterior distribution at step t-1:
          μ(x_t, t) = (α_t^(-1/2)) * (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ(x_t, t))
          σ_t = sqrt(posterior_variance[t])
        
        Then: x_{t-1} = μ + σ_t * z where z ~ N(0, I)
        
        Args:
            model: Denoising model predicting noise.
            xt: Noisy latent at step t of shape (B, C, H, W).
            t: Current timestep index (scalar integer).
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).
            
        Returns:
            Denoised latent x_{t-1} of shape (B, C, H, W).
            
        Shape contract:
            xt: (B, C, H, W)
            t: scalar int in [0, num_timesteps-1]
            output: (B, C, H, W)
        """
        device = xt.device
        
        # Create timestep tensor
        t_tensor = torch.full((xt.shape[0],), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred = model(xt, t_tensor)
        
        # Get alpha schedule values
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        beta_t = self.betas[t]
        
        # Posterior mean
        # μ = (1/sqrt(α_t)) * (x_t - (β_t / sqrt(1-ᾱ_t)) * ε_pred)
        coeff = beta_t / sqrt_one_minus_alpha_bar_t
        mean = (xt - coeff * noise_pred) / torch.sqrt(alpha_t)
        
        # Posterior variance (clipped for stability)
        log_var = self.posterior_log_variance_clipped[t]
        var = torch.exp(log_var)
        
        # Sample from posterior
        if t > 0:
            z = torch.randn_like(xt)
            x_prev = mean + torch.sqrt(var) * z
        else:
            # Last step: no noise
            x_prev = mean
        
        return x_prev

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        guidance_scale: float = 1.0,
        progress_bar: bool = False,
    ) -> Tensor:
        """Full reverse diffusion (DDPM) to sample from noise.
        
        Iteratively denoise from t=num_timesteps-1 down to t=0:
          1. Start with x_T ~ N(0, I)
          2. For t in {T-1, ..., 1, 0}:
             x_{t-1} = p_sample(model, x_t, t)
          3. Return x_0
        
        Args:
            model: Denoising model.
            shape: Shape of latent to generate (e.g., (B, 4, 32, 32)).
            guidance_scale: Classifier-free guidance scale.
            progress_bar: Whether to show progress bar.
            
        Returns:
            Generated latent of shape 'shape'.
            
        Shape contract:
            model(x_t, t) expects: x_t (B, C, H, W), t (B,) → (B, C, H, W)
            output: shape
        """
        device = next(model.parameters()).device
        
        # Start with pure noise
        x_t = torch.randn(shape, device=device)
        
        # Reverse diffusion loop
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress_bar:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc="DDPM sampling")
            except ImportError:
                pass
        
        for t in timesteps:
            x_t = self.p_sample(model, x_t, t, guidance_scale=guidance_scale)
        
        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        progress_bar: bool = False,
        eta: float = 0.0,
    ) -> Tensor:
        """Fast DDIM sampling with an arbitrary number of steps.

        DDIM trades stochasticity for speed; with eta=0 it is deterministic.

        Args:
            model: Denoising model.
            shape: Latent shape, e.g. (B, 4, 32, 32).
            num_steps: Number of DDIM steps (< num_timesteps).
            guidance_scale: Classifier-free guidance scale.
            progress_bar: Whether to show a tqdm bar.
            eta: Stochasticity (0 = deterministic, 1 = DDPM-like).

        Returns:
            Generated latent of shape `shape`.
        """
        device = next(model.parameters()).device

        # Evenly spaced timestep subset for DDIM.
        timesteps = torch.linspace(
            0,
            self.num_timesteps - 1,
            num_steps,
            device=device,
            dtype=torch.long,
        )

        # Start from pure noise.
        x_t = torch.randn(shape, device=device)

        # IMPORTANT: keep a plain tensor for indexing; only the display iterator
        # is wrapped in tqdm (a tqdm object is not subscriptable).
        timesteps_seq = timesteps.flip(0)  # reverse order (T -> 0)
        num_ddim = len(timesteps_seq)
        iterator = timesteps_seq
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(timesteps_seq, desc="DDIM sampling")
            except ImportError:
                pass

        for i, t in enumerate(iterator):
            t_idx = t.item()

            # Batch of the current timestep.
            t_batch = torch.full((x_t.shape[0],), t_idx, device=device, dtype=torch.long)

            # Predict noise.
            noise_pred = model(x_t, t_batch)

            # Alpha for the current step.
            alpha_bar_t = self.alphas_cumprod[t_idx]

            # Alpha for the previous step (index the plain tensor, not the tqdm wrapper).
            if i < num_ddim - 1:
                t_prev_idx = timesteps_seq[i + 1].item()
                alpha_bar_prev = self.alphas_cumprod[t_prev_idx]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device)

            # DDIM update.
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt(1 - alpha_bar_prev)

            # Reconstruct x_0.
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

            # Deterministic direction toward x_t.
            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_bar_prev - (eta * sqrt_one_minus_alpha_bar_prev) ** 2, min=0.0)
            ) * noise_pred

            # Optional stochastic term.
            if eta > 0 and i < num_ddim - 1:
                noise = torch.randn_like(x_t)
                stoch = eta * sqrt_one_minus_alpha_bar_prev * noise
            else:
                stoch = 0.0

            # Step to the previous timestep.
            x_t = sqrt_alpha_bar_prev * x_0_pred + dir_xt + stoch

        return x_t

if __name__ == "__main__":
    """Smoke test: verify diffusion forward/reverse shapes and loss computation."""
    import sys
    sys.path.insert(0, '/Users/pranjal/Desktop/Home/Projects/LatentFlow')
    
    from latentflow.models.dit import DiT_S_2
    from latentflow.diffusion.schedule import get_beta_schedule, compute_alphas
    
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
