"""VAE wrapper for encoding/decoding images to/from latent space.

Uses a pretrained Stable Diffusion VAE from HuggingFace.
Frozen, eval mode, no gradients.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """Wraps a pretrained VAE for encoding/decoding.
    
    The VAE is loaded frozen in eval mode with no_grad context.
    Encoding reduces spatial dimensions by 8x and applies latent scaling.
    
    Args:
        model_id: HuggingFace model ID (e.g., "stabilityai/sd-vae-ft-ema").
        latent_scale_factor: Scaling factor for latents (typically 0.18215).
    """

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-ema",
        latent_scale_factor: float = 0.18215,
    ) -> None:
        """Initialize VAE wrapper."""
        super().__init__()
        self.model_id = model_id
        self.latent_scale_factor = latent_scale_factor
        
        # Load pretrained VAE
        self.vae = AutoencoderKL.from_pretrained(model_id)
        
        # Freeze VAE; we only use it for encoding/decoding, not training
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Set to eval mode permanently
        self.vae.eval()

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Encode images to latent space.
        
        Math: z = scale_factor * z_raw, where z_raw ~ N(0, 1) from VAE encoder.
        Spatial dimensions reduce by 8x: [B, 3, H, W] -> [B, 4, H/8, W/8].
        
        Args:
            x: Image tensor of shape (B, 3, H, W) with values in [-1, 1].
            
        Returns:
            Latent tensor of shape (B, 4, H/8, W/8), scaled by latent_scale_factor.
        """
        # Encode to latent distribution
        posterior = self.vae.encode(x).latent_dist
        # Sample from distribution (or use mean for determinism)
        z_raw = posterior.sample()
        # Apply scaling
        z = z_raw * self.latent_scale_factor
        return z

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        """Decode latents to image space.
        
        Math: x = VAE_decoder(z_raw), where z_raw = z / scale_factor.
        Spatial dimensions expand by 8x: [B, 4, H/8, W/8] -> [B, 3, H, W].
        
        Args:
            z: Latent tensor of shape (B, 4, H/8, W/8), scaled by latent_scale_factor.
            
        Returns:
            Image tensor of shape (B, 3, H, W) with values in [-1, 1].
        """
        # Undo scaling
        z_raw = z / self.latent_scale_factor
        # Decode
        x = self.vae.decode(z_raw).sample
        return x

    @property
    def device(self) -> torch.device:
        """Return device of VAE."""
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of VAE."""
        return next(self.vae.parameters()).dtype
