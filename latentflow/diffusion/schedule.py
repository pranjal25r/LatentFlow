"""Beta schedules for diffusion process.

References:
  DDPM: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
  Improved DDPM: Nichol & Dhariwal (2021)
"""

from typing import Tuple
import numpy as np
import torch
from torch import Tensor


def linear_schedule(
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tensor:
    """Linear beta schedule.
    
    Beta values increase linearly from beta_start to beta_end over timesteps.
    Formula: beta_t = beta_start + t/T * (beta_end - beta_start)
    
    Args:
        num_timesteps: Number of diffusion steps.
        beta_start: Starting beta value (typically 1e-4).
        beta_end: Ending beta value (typically 0.02).
        
    Returns:
        Betas of shape (num_timesteps,) with values in [beta_start, beta_end].
    """
    # Linear interpolation from beta_start to beta_end
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    return betas


def cosine_schedule(
    num_timesteps: int,
    s: float = 0.008,
) -> Tensor:
    """Cosine beta schedule (improved DDPM).
    
    Provides smoother noise scheduling using cosine annealing.
    First computes alpha_bar using cosine, then derives betas.
    
    Formula:
        f(t) = cos((t/T + s)/(1 + s) * π/2)²
        alpha_bar_t = f(t) / f(0)
        beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    
    Args:
        num_timesteps: Number of diffusion steps.
        s: Small offset for numerical stability (typically 0.008).
        
    Returns:
        Betas of shape (num_timesteps,).
    """
    # Compute alpha_bar values using cosine
    # t ranges from [0, num_timesteps]
    t = torch.arange(0, num_timesteps + 1, dtype=torch.float32)
    
    # Cosine annealing with offset s
    alpha_bar = torch.cos(
        (t / num_timesteps + s) / (1 + s) * np.pi / 2
    ) ** 2
    
    # Normalize so alpha_bar[0] = 1.0
    alpha_bar = alpha_bar / alpha_bar[0]
    
    # Clamp to [1e-5, 0.999] to avoid numerical issues
    alpha_bar = torch.clamp(alpha_bar, min=1e-5, max=0.999)
    
    # Derive betas from consecutive alpha_bar values
    # beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    
    # Clamp betas to [0, 0.999]
    betas = torch.clamp(betas, min=0, max=0.999)
    
    return betas


def get_beta_schedule(
    schedule: str,
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> Tensor:
    """Get beta schedule by name.
    
    Args:
        schedule: Schedule type ("linear" or "cosine").
        num_timesteps: Number of diffusion steps.
        beta_start: Starting beta (for linear schedule).
        beta_end: Ending beta (for linear schedule).
        
    Returns:
        Betas of shape (num_timesteps,).
    """
    if schedule == "linear":
        return linear_schedule(num_timesteps, beta_start, beta_end)
    elif schedule == "cosine":
        return cosine_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def compute_alphas(betas: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute alpha quantities needed for q_sample and reverse process.
    
    Given betas, compute:
      alpha_t = 1 - beta_t
      alpha_bar_t = prod(alpha_s) for s in [0, t] (cumulative product)
      sqrt_alpha_bar_t = sqrt(alpha_bar_t)
      sqrt_one_minus_alpha_bar_t = sqrt(1 - alpha_bar_t)
      
    These quantities are used in:
      - q_sample: x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
      - p_sample: For reverse process coefficients
    
    Args:
        betas: Betas of shape (num_timesteps,).
        
    Returns:
        Tuple of:
          - alphas: alpha_t = 1 - beta_t
          - alphas_cumprod: alpha_bar_t (cumulative product)
          - sqrt_alphas_cumprod: sqrt(alpha_bar_t)
          - sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t)
          - alphas_cumprod_prev: alpha_bar_{t-1} (shifted by 1, padded with 1.0)
          - posterior_variance: For reverse posterior sampling
    """
    # Alphas: 1 - betas
    alphas = 1.0 - betas  # [num_timesteps]
    
    # Cumulative product of alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # [num_timesteps]
    
    # Quantities for q_sample
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # Previous timestep's alpha_cumprod (for posterior)
    # Prepend 1.0 at the beginning, remove last element
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]]
    )
    
    # Posterior variance for p_sample reverse process
    # variance = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
    posterior_variance = (
        (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * betas
    )
    
    # Clamp to avoid numerical issues
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    
    return (
        alphas,
        alphas_cumprod,
        sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod,
        alphas_cumprod_prev,
        posterior_variance,
    )
