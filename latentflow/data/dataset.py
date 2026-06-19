"""Datasets for CelebA-HQ images and precomputed latents.

References:
  CelebA-HQ: Karras et al., "Progressive Growing of GANs" (2018)
"""

from typing import Optional, Tuple, Union
import os
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebAHQDataset(Dataset):
    """CelebA-HQ dataset (images).
    
    Loads 256×256 (or custom resolution) face images. Normalizes to [-1, 1].
    Optionally encodes on-the-fly to latents (slow; use CachedLatentDataset in production).
    
    Args:
        data_dir: Path to CelebA-HQ directory (must contain image files).
        resolution: Resolution to load and resize to (default 256).
        vae: Optional VAE wrapper for on-the-fly encoding to latents.
    
    Shape contract:
      Output image: [3, resolution, resolution] in [-1, 1]
      Output latent (if vae provided): [4, resolution/8, resolution/8] scaled
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 256,
        vae: Optional[object] = None,
    ) -> None:
        """Initialize CelebA-HQ dataset."""
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.vae = vae

        # Scan directory for image files
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(sorted(self.data_dir.glob(ext)))
            self.image_paths.extend(sorted(self.data_dir.glob(f'*/{ext}')))
        
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {data_dir}")
        
        # Remove duplicates and sort
        self.image_paths = sorted(set(self.image_paths))

        # Normalization to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Maps [0, 1] -> [-1, 1]
        ])

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tensor:
        """Load and return a sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            If vae is None: Image tensor of shape (3, resolution, resolution).
            If vae is provided: Latent tensor of shape (4, resolution/8, resolution/8).
        """
        img_path = self.image_paths[idx]
        
        # Load and transform image
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        
        # Optionally encode to latent
        if self.vae is not None:
            # Add batch dimension, encode, remove batch
            x = x.unsqueeze(0)  # [1, 3, H, W]
            z = self.vae.encode(x)  # [1, 4, H/8, W/8]
            x = z.squeeze(0)  # [4, H/8, W/8]
        
        return x


class CachedLatentDataset(Dataset):
    """Precomputed latent dataset (cached VAE encodings).
    
    Loads precomputed latents from disk as individual .pt files.
    This avoids re-running the VAE encoder during training (huge speedup).
    
    Assumes latent files are named consistently: 00000.pt, 00001.pt, etc.
    Or any .pt files in the directory.
    
    Args:
        latent_dir: Path to cached latent directory.
    
    Shape contract:
      Each latent: [4, H, W] where H, W depend on original image resolution
      E.g., 256×256 image -> [4, 32, 32] latent
    """

    def __init__(self, latent_dir: str) -> None:
        """Initialize latent dataset."""
        self.latent_dir = Path(latent_dir)

        # Scan directory for .pt files
        self.latent_paths = sorted(self.latent_dir.glob('*.pt'))
        
        if not self.latent_paths:
            raise FileNotFoundError(f"No .pt files found in {latent_dir}")

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.latent_paths)

    def __getitem__(self, idx: int) -> Tensor:
        """Load and return a precomputed latent.
        
        Args:
            idx: Sample index.
            
        Returns:
            Latent tensor of shape (4, H, W).
        """
        latent_path = self.latent_paths[idx]
        z = torch.load(latent_path, weights_only=True)  # weights_only for security
        return z
