"""Seeding utilities for reproducibility.

Sets random seeds for Python, NumPy, PyTorch (CPU and CUDA).
Also sets deterministic flags.
"""

from typing import Optional
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Configures:
      - Python's random module
      - NumPy's random generator
      - PyTorch (CPU)
      - PyTorch CUDA (if available)
      - Deterministic flags for reproducibility
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Enable deterministic algorithms (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for determinism
