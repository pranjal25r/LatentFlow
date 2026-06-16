"""Diffusion Transformer (DiT) model components.

References:
  Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)
  https://arxiv.org/abs/2212.09748
"""

from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply modulation: (x * (1 + scale)) + shift.
    
    Used in adaLN-Zero conditioning. Allows timestep embedding to control
    the layer norm output.
    
    Args:
        x: Normalized input of shape (B, ..., D).
        shift: Shift parameter of shape (B, D) or broadcastable.
        scale: Scale parameter of shape (B, D) or broadcastable.
        
    Returns:
        Modulated tensor of same shape as x.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """Embeds latent patches into tokens with positional encoding.
    
    Converts latent space [B, C, H, W] into a sequence of patch tokens.
    Each patch is projected to hidden_size, then summed with positional embeddings.
    
    Args:
        latent_channels: Number of input latent channels (e.g., 4).
        hidden_size: Hidden/embedding dimension.
        patch_size: Spatial patch size in latent space (e.g., 2).
        use_sincos_pos: If True, use sinusoidal positional embeddings; else learnable.
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_size: int,
        patch_size: int = 2,
        use_sincos_pos: bool = True,
    ) -> None:
        """Initialize patch embedding layer."""
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.use_sincos_pos = use_sincos_pos
        
        # Patchify via 2D convolution
        # Input: [B, C, H, W] -> Output: [B, hidden_size, H/patch_size, W/patch_size]
        self.proj = nn.Conv2d(
            latent_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # Calculate number of patches
        # Assumes latent is square; for 256x256 image -> 32x32 latent, patch_size=2 -> 16x16 patches
        # We'll compute this dynamically or store num_patches later
        self.num_patches: Optional[int] = None

    def forward(self, x: Tensor) -> Tensor:
        """Embed patches and return flattened sequence with position encoding.
        
        Args:
            x: Latent tensor of shape (B, C, H, W).
            
        Returns:
            Token sequence of shape (B, num_patches, hidden_size).
            num_patches = (H / patch_size) * (W / patch_size).
        """
        # Patchify: [B, C, H, W] -> [B, hidden_size, H/patch_size, W/patch_size]
        x = self.proj(x)
        B, C, H, W = x.shape
        
        # Store num_patches for later use
        self.num_patches = H * W
        
        # Flatten spatial dims: [B, hidden_size, H, W] -> [B, hidden_size, num_patches]
        x = x.flatten(2)  # [B, hidden_size, H*W]
        
        # Transpose to [B, num_patches, hidden_size]
        x = x.transpose(1, 2)
        
        # Add positional embeddings
        pos_emb = self._get_pos_emb(B, H, W, device=x.device, dtype=x.dtype)
        x = x + pos_emb
        
        return x

    def _get_pos_emb(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Get positional embeddings for patch grid.
        
        Args:
            B: Batch size.
            H: Patch grid height.
            W: Patch grid width.
            device: Device to place embeddings on.
            dtype: Data type.
            
        Returns:
            Positional embeddings of shape (B, H*W, hidden_size).
        """
        if self.use_sincos_pos:
            # Sinusoidal positional embeddings (2D)
            pos_emb = self._sincos_pos_emb_2d(H, W, self.hidden_size, device, dtype)
        else:
            # Learnable positional embeddings
            # Lazily initialize once when we know the grid size
            if not hasattr(self, 'pos_emb_table'):
                self.pos_emb_table = nn.Parameter(
                    torch.randn(1, H * W, self.hidden_size, device=device, dtype=dtype)
                )
            pos_emb = self.pos_emb_table
        
        # Expand to batch size: [1, H*W, hidden_size] -> [B, H*W, hidden_size]
        pos_emb = pos_emb.expand(B, -1, -1)
        return pos_emb

    @staticmethod
    def _sincos_pos_emb_2d(
        H: int,
        W: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """2D sinusoidal positional embeddings.
        
        Args:
            H: Grid height.
            W: Grid width.
            dim: Embedding dimension.
            device: Device.
            dtype: Data type.
            
        Returns:
            Positional embeddings of shape (1, H*W, dim).
        """
        # Create frequency bands for sinusoidal encoding
        d_model = dim
        
        # 1D sinusoidal embeddings for height
        pe_h = torch.zeros(H, d_model // 2, device=device, dtype=dtype)
        position_h = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2, device=device, dtype=dtype) *
            -(math.log(10000.0) / (d_model // 2))
        )
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)
        
        # 1D sinusoidal embeddings for width
        pe_w = torch.zeros(W, d_model // 2, device=device, dtype=dtype)
        position_w = torch.arange(W, device=device, dtype=dtype).unsqueeze(1)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)
        
        # 2D positional embedding: concatenate H and W embeddings
        # For each (i, j) position, concatenate pe_h[i] + pe_w[j]
        pe_2d = []
        for i in range(H):
            for j in range(W):
                pe_2d.append(torch.cat([pe_h[i], pe_w[j]]))
        
        pe_2d = torch.stack(pe_2d, dim=0)  # [H*W, d_model]
        pe_2d = pe_2d.unsqueeze(0)  # [1, H*W, d_model]
        
        return pe_2d


class TimestepEmbedder(nn.Module):
    """Embeds diffusion timesteps into vectors via sinusoidal encoding + MLP.
    
    Produces a conditioning vector from timestep indices. This is used to
    modulate the layer norms in each DiT block (adaLN-Zero).
    
    Args:
        hidden_size: Output embedding dimension and intermediate MLP size.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialize timestep embedder."""
        super().__init__()
        self.hidden_size = hidden_size
        
        # Sinusoidal embedding dimension (before MLP)
        self.time_dim = hidden_size
        
        # MLP to project sinusoidal embeddings to conditioning vector
        # Input: time_dim (sinusoidal) -> Output: hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Embed timesteps.
        
        Args:
            t: Timestep indices of shape (B,) with values in [0, num_timesteps).
            
        Returns:
            Time embeddings of shape (B, hidden_size).
        """
        # Sinusoidal positional encoding for timesteps
        # Similar to Transformer positional encoding but applied to scalar timesteps
        
        # t: [B] -> expand to [B, time_dim]
        B = t.shape[0]
        device = t.device
        dtype = t.dtype
        
        # Frequency bands for sinusoidal encoding
        freqs = torch.exp(
            torch.arange(0, self.time_dim, 2, device=device, dtype=dtype) *
            -(math.log(10000.0) / self.time_dim)
        )
        
        # t_scaled: [B] -> [B, 1] -> [B, time_dim//2]
        t_scaled = t.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Interleave sin and cos
        # sin: [B, time_dim//2], cos: [B, time_dim//2]
        t_sin = torch.sin(t_scaled)
        t_cos = torch.cos(t_scaled)
        
        # Concatenate: [B, time_dim]
        t_emb = torch.cat([t_sin, t_cos], dim=1)
        
        # Project through MLP to conditioning vector
        t_emb = self.mlp(t_emb)  # [B, hidden_size]
        
        return t_emb


class DiTBlock(nn.Module):
    """Single Transformer block for DiT with adaLN-Zero conditioning.
    
    Combines self-attention, MLP, and adaptive layer normalization.
    The timestep embedding produces scale/shift parameters that modulate
    the layer norms (adaLN-Zero).
    
    Args:
        hidden_size: Hidden/embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of mlp hidden size to hidden size (e.g., 4.0).
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing for memory.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        """Initialize DiT block."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        
        # Pre-norm layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Multi-head self-attention
        # Q, K, V projections + output projection
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP block: hidden_size -> hidden_size*mlp_ratio -> hidden_size
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        
        # adaLN-Zero parameters
        # Linear layer to project timestep embedding to scale/shift/gate params
        # We need 6 parameters total: 2 for attention norm (scale, shift), 
        # 1 for attention gate, 2 for MLP norm, 1 for MLP gate
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(
        self,
        x: Tensor,
        t_emb: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with adaptive layer normalization.
        
        Args:
            x: Token sequence of shape (B, L, hidden_size).
            t_emb: Timestep embeddings of shape (B, hidden_size).
            attn_mask: Optional attention mask of shape (L, L).
            
        Returns:
            Updated token sequence of shape (B, L, hidden_size).
        """
        # Extract adaLN parameters from timestep embedding
        # Output: [B, 6*hidden_size]
        adaLN_params = self.adaLN_mlp(t_emb)
        
        # Split into attention and MLP params
        # [B, 6*hidden_size] -> 6 tensors of [B, hidden_size]
        (
            attn_scale, attn_shift, attn_gate,
            mlp_scale, mlp_shift, mlp_gate,
        ) = adaLN_params.chunk(6, dim=1)
        
        # Self-attention with adaLN-Zero
        # Normalize and modulate input
        x_norm = self.norm1(x)  # [B, L, hidden_size]
        x_norm = modulate(x_norm, attn_shift, attn_scale)  # [B, L, hidden_size]
        
        # Apply attention
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=attn_mask,
            need_weights=False,
        )  # [B, L, hidden_size]
        
        # Gated residual: x = x + gate * attn_out
        # Gate is [B, hidden_size], unsqueeze to [B, 1, hidden_size] for broadcasting
        x = x + attn_gate.unsqueeze(1) * attn_out
        
        # MLP with adaLN-Zero
        # Normalize and modulate input
        x_norm = self.norm2(x)  # [B, L, hidden_size]
        x_norm = modulate(x_norm, mlp_shift, mlp_scale)  # [B, L, hidden_size]
        
        # Apply MLP
        mlp_out = self.mlp(x_norm)  # [B, L, hidden_size]
        
        # Gated residual: x = x + gate * mlp_out
        # Gate is [B, hidden_size], unsqueeze to [B, 1, hidden_size] for broadcasting
        x = x + mlp_gate.unsqueeze(1) * mlp_out
        
        return x


class FinalLayer(nn.Module):
    """Final output layer for DiT with adaLN modulation.
    
    Converts token sequence back to latent shape, predicting noise
    (or sample, depending on prediction_type config).
    
    Args:
        hidden_size: Hidden/embedding dimension.
        latent_channels: Number of latent channels to predict noise for.
        patch_size: Patch size used during embedding.
        prediction_type: Type of prediction ("epsilon", "sample", "v_prediction").
    """

    def __init__(
        self,
        hidden_size: int,
        latent_channels: int,
        patch_size: int = 2,
        prediction_type: str = "epsilon",
    ) -> None:
        """Initialize final layer."""
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.prediction_type = prediction_type
        
        # Output channels: latent_channels * patch_size^2 (one value per sub-patch)
        self.out_channels = latent_channels * (patch_size ** 2)
        
        # Pre-norm
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # adaLN parameters: scale and shift
        self.adaLN_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        
        # Output projection: hidden_size -> out_channels
        self.linear = nn.Linear(hidden_size, self.out_channels)

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """Predict noise from token sequence with adaLN modulation.
        
        Args:
            x: Token sequence of shape (B, num_patches, hidden_size).
            t_emb: Timestep embeddings of shape (B, hidden_size).
            
        Returns:
            Noise prediction of shape (B, latent_channels, H/patch_size, W/patch_size).
        """
        # Extract adaLN parameters
        adaLN_params = self.adaLN_mlp(t_emb)  # [B, 2*hidden_size]
        scale, shift = adaLN_params.chunk(2, dim=1)  # 2x [B, hidden_size]
        
        # Apply adaptive layer norm
        x = self.norm(x)  # [B, num_patches, hidden_size]
        x = modulate(x, shift, scale)  # [B, num_patches, hidden_size]
        
        # Project to output channels
        x = self.linear(x)  # [B, num_patches, out_channels]
        
        # Unpatchify: [B, num_patches, out_channels] -> [B, latent_channels, H, W]
        # We need to reshape assuming square grid
        num_patches = x.shape[1]
        H = W = int(math.sqrt(num_patches))  # Grid size (assuming square)
        
        # Reshape: [B, H*W, latent_channels*patch_size^2]
        x = x.reshape(
            x.shape[0],
            H, W,
            self.latent_channels,
            self.patch_size,
            self.patch_size,
        )
        
        # Rearrange: [B, H, W, C, patch_size, patch_size]
        # -> [B, C, H*patch_size, W*patch_size]
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, C, H, patch_size, W, patch_size]
        x = x.reshape(
            x.shape[0],
            self.latent_channels,
            H * self.patch_size,
            W * self.patch_size,
        )
        
        return x


class DiT(nn.Module):
    """Diffusion Transformer for latent-space image generation.
    
    Architecture:
      1. Patch embed latent → tokens [B, num_patches, hidden_size]
      2. Embed timestep → conditioning [B, hidden_size]
      3. Stack of DiT blocks with adaLN conditioning
      4. Final layer → noise prediction [B, latent_channels, H, W]
    
    Args:
        latent_channels: Number of VAE latent channels (typically 4).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden size ratio for MLPs.
        patch_size: Spatial patch size in latent space.
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing for memory.
        prediction_type: Type of prediction ("epsilon", "sample", "v_prediction").
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        prediction_type: str = "epsilon",
    ) -> None:
        """Initialize DiT model."""
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.prediction_type = prediction_type

        # Patch embedder: [B, 4, 32, 32] -> [B, 256, hidden_size]
        self.patch_embed = PatchEmbed(
            latent_channels=latent_channels,
            hidden_size=hidden_size,
            patch_size=patch_size,
            use_sincos_pos=True,
        )
        
        # Timestep embedder: [B] -> [B, hidden_size]
        self.time_embed = TimestepEmbedder(hidden_size)
        
        # Stack of DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(depth)
        ])
        
        # Final output layer
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            latent_channels=latent_channels,
            patch_size=patch_size,
            prediction_type=prediction_type,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Predict noise for diffusion step.
        
        Args:
            x: Noisy latent tensor of shape (B, C, H, W).
              e.g., (batch_size, 4, 32, 32) for 256x256 images
            t: Timestep indices of shape (B,) with values in [0, num_timesteps).
            
        Returns:
            Noise prediction of shape (B, C, H, W), same as input.
        """
        # Input shape: [B, latent_channels, H, W]
        B = x.shape[0]
        
        # Patchify and add positional embeddings
        # [B, C, H, W] -> [B, num_patches, hidden_size]
        x_tokens = self.patch_embed(x)
        
        # Embed timesteps
        # [B] -> [B, hidden_size]
        t_emb = self.time_embed(t)
        
        # Apply DiT blocks
        for block in self.blocks:
            # Each block: [B, num_patches, hidden_size] + [B, hidden_size]
            # -> [B, num_patches, hidden_size]
            x_tokens = block(x_tokens, t_emb)
        
        # Final layer predicts noise
        # [B, num_patches, hidden_size] + [B, hidden_size]
        # -> [B, latent_channels, H, W]
        noise_pred = self.final_layer(x_tokens, t_emb)
        
        return noise_pred

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained_config(cls, config: Dict[str, Any]) -> "DiT":
        """Create DiT from config dictionary.
        
        Args:
            config: Configuration dictionary (typically loaded from YAML).
            
        Returns:
            DiT instance with config parameters.
        """
        return cls(
            latent_channels=config.get('latent_channels', 4),
            hidden_size=config.get('dit', {}).get('hidden_size', 768),
            depth=config.get('dit', {}).get('depth', 12),
            num_heads=config.get('dit', {}).get('num_heads', 12),
            mlp_ratio=config.get('dit', {}).get('mlp_ratio', 4.0),
            patch_size=config.get('dit', {}).get('patch_size', 2),
            dropout=config.get('dit', {}).get('dropout', 0.0),
            use_checkpoint=config.get('dit', {}).get('use_checkpoint', False),
            prediction_type=config.get('diffusion', {}).get('prediction_type', 'epsilon'),
        )


# Named presets for DiT models
def DiT_S_2(**kwargs) -> DiT:
    """Small DiT with patch_size=2.
    
    Parameters: ~97M
    """
    return DiT(
        hidden_size=384,
        depth=12,
        num_heads=6,
        patch_size=2,
        **kwargs,
    )


def DiT_B_2(**kwargs) -> DiT:
    """Base DiT with patch_size=2.
    
    Parameters: ~312M
    """
    return DiT(
        hidden_size=768,
        depth=12,
        num_heads=12,
        patch_size=2,
        **kwargs,
    )


if __name__ == "__main__":
    """Smoke test: instantiate DiT-S/2 and run forward pass."""
    print("=" * 80)
    print("DiT Smoke Test")
    print("=" * 80)
    
    # Create small DiT model
    model = DiT_S_2(latent_channels=4, dropout=0.0)
    print(f"\n✓ Created DiT-S/2 model")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  Depth: {model.depth}")
    print(f"  Num heads: {model.num_heads}")
    print(f"  Total parameters: {model.num_parameters():,}")
    
    # Create sample input
    batch_size = 4
    latent_channels = 4
    latent_height = 32
    latent_width = 32
    num_timesteps = 1000
    
    # Random latent tensor: [B, C, H, W]
    # Represents 4 noisy 256x256 images compressed to 32x32 latent space
    latent = torch.randn(batch_size, latent_channels, latent_height, latent_width)
    
    # Random timesteps: [B] with values in [0, num_timesteps)
    timesteps = torch.randint(0, num_timesteps, (batch_size,))
    
    print(f"\n✓ Created sample inputs")
    print(f"  Latent shape: {tuple(latent.shape)}")
    print(f"  Timesteps shape: {tuple(timesteps.shape)}")
    
    # Forward pass
    with torch.no_grad():
        noise_pred = model(latent, timesteps)
    
    print(f"\n✓ Forward pass successful")
    print(f"  Output shape: {tuple(noise_pred.shape)}")
    
    # Verify output shape matches input
    assert noise_pred.shape == latent.shape, (
        f"Output shape {noise_pred.shape} != input shape {latent.shape}"
    )
    print(f"✓ Output shape matches input shape")
    
    # Test with DiT-B/2 (larger model)
    print(f"\n" + "=" * 80)
    print("Testing DiT-B/2 (larger model)")
    print("=" * 80)
    
    model_b = DiT_B_2(latent_channels=4, dropout=0.0)
    print(f"\n✓ Created DiT-B/2 model")
    print(f"  Total parameters: {model_b.num_parameters():,}")
    
    with torch.no_grad():
        noise_pred_b = model_b(latent, timesteps)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {tuple(noise_pred_b.shape)}")
    assert noise_pred_b.shape == latent.shape
    print(f"✓ Output shape matches input shape")
    
    print(f"\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
